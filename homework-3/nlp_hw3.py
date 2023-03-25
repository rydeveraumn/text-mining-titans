# third party
import evaluate
import numpy as np
import pandas as pd
import torch
import copy
from datasets import load_dataset, load_metric
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

def greedy_search_generate(model, input_ids, max_length):
    is_enc_dec = hasattr(model, 'encoder') and hasattr(model, 'decoder')
    if is_enc_dec:
        gen_text = torch.tensor([[0]], dtype=torch.int32)
        for i in range(max_length):
            logits = model.forward(input_ids,decoder_input_ids=gen_text).logits[0,-1,:]
            gen_text = torch.cat((gen_text,torch.argmax(logits).unsqueeze(0).unsqueeze(0)),dim=1)
            if gen_text[0][-1] == model.generation_config.eos_token_id:
                break
    else:
        gen_text = copy.deepcopy(input_ids)
        for i in range(max_length - len(input_ids[0])):
            logits = model.forward(gen_text).logits[0,-1,:]
            gen_text = torch.cat((gen_text,torch.argmax(logits).unsqueeze(0).unsqueeze(0)),dim=1)
            if gen_text[0][-1] == model.generation_config.eos_token_id:
                break
    return gen_text

def calc_perplexity(model, sentence):
    """
    Reference: https://huggingface.co/docs/transformers/perplexity
    """

    max_length = model.config.n_positions
    stride = 512
    seq_len = sentence.size(0)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = sentence[begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    nll = torch.stack(nlls).sum() / end_loc
    ppl = torch.exp(nll)
    return nll, ppl


def calc_stats(mname, model, tokenizer, outputs, perplexity=True):
    if perplexity == False:
        rand_idx = np.random.randint(len(outputs))
        return {"responses": outputs, "nll": 0, "ppl": 0, "ppl_argmin": rand_idx}, None

    nlls = []
    ppls = []
    full_results_list = []
    # Just add the results here instead
    print(mname + ":\n" + 100 * "-")
    for i, output in enumerate(outputs):
        # We want to store the full results
        full_results = {}

        decoded_outputs = tokenizer.decode(output, skip_special_tokens=True)
        print("{}: {}".format(i, decoded_outputs))
        nll, ppl = calc_perplexity(model, output)
        nlls.append(nll)
        ppls.append(ppl)
        print(
            100 * "-"
            + "\n"
            + "NLL: {}\tPerplexity: {}".format(np.mean(nlls), np.mean(ppls))
        )
        print(100 * "-" + "\n")

        # Add to results
        full_results[f"reponse"] = decoded_outputs
        full_results[f"nll"] = nll.numpy()
        full_results[f"ppl"] = ppl.numpy()
        full_results["algorithm_name"] = mname

        full_results_list.append(full_results)

    # Compute the results
    results = {
        "responses": outputs,
        "nll": np.min(nlls),
        "ppl": np.min(ppls),
        "ppl_argmin": np.argmin(ppls),
    }

    return results, full_results_list


def get_outputs(
    model,
    tokenizer,
    input_ids,
    strategy="greedy",
    perplexity=True,
    max_length=50,
    num_return_sequences=5,
    top_k=10,
    top_p=0.98,
    scratch = True
):
    out_json = None
    if strategy == "greedy":
        # greedy search
        if scratch:
            greedy_output = greedy_search_generate(model, input_ids, max_length)
        else:
            greedy_output = model.generate(input_ids, max_length=50)
        out_json, full_results_list = calc_stats(
            "Greedy Output", model, tokenizer, greedy_output, perplexity=perplexity
        )

    elif strategy == "beam":
        # beam search
        beam_outputs = model.generate(
            input_ids,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )
        out_json, full_results_list = calc_stats(
            "Beam Search Output", model, tokenizer, beam_outputs, perplexity=perplexity
        )

    elif strategy == "sampling":  # random sampling with temperature
        sample_outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_k=0,
            temperature=0.7,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )
        out_json, full_results_list = calc_stats(
            "Random Sampling with temp Output",
            model,
            tokenizer,
            sample_outputs,
            perplexity=perplexity,
        )

    elif strategy == "top_k":  # top-k sampling
        topk_outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )
        out_json, full_results_list = calc_stats(
            "Top-k Output", model, tokenizer, topk_outputs, perplexity=perplexity
        )

    elif strategy == "top_p":  # top-p sampling
        topp_outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_p=top_p,
            top_k=0,  # has to be zero
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )
        out_json, full_results_list = calc_stats(
            "Top-p Output", model, tokenizer, topp_outputs, perplexity=perplexity
        )

    elif strategy == "topp_topk":  # top-p, top_k sampling
        topp_topk_outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )
        out_json, full_results_list = calc_stats(
            "Top-p, Top-k Output",
            model,
            tokenizer,
            topp_topk_outputs,
            perplexity=perplexity,
        )

    return out_json, full_results_list


def get_paraphrase(
    sentence,
    model,
    tokenizer,
    prefix="paraphrase: ",
    strategy="greedy",
    n_predictions=5,
    top_k=5,
    top_p=0.80,
    max_length=50,
    device="cpu",
):
    """
    Function that will generate paraphrases
    """
    text = prefix + sentence
    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)

    out_json, full_results_list = get_outputs(
        model,
        tokenizer,
        input_ids,
        strategy=strategy,
        perplexity=False,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=n_predictions,
    )
    best_ppl_idx = out_json["ppl_argmin"]
    best_sent = out_json["responses"][best_ppl_idx]
    generated_sent = tokenizer.decode(
        best_sent, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return generated_sent


def run_tasks():
    """
    We will use this function to run the different tasks for HW3
    """
    # Task: 1 is the implementation of decoding algorithms
    # Set up the tokenizer
    print("Generating outputs for task 1 ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up the model
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )
    input_ids = tokenizer.encode(
        "Men are intelligent and women are", return_tensors="pt"
    )

    outputs = []
    methods = ["greedy", "beam", "sampling", "top_k", "top_p", "topp_topk"]
    for method in methods:
        if method in ["greedy", "beam", "sampling"]:
            out_json, full_results_list = get_outputs(
                model, tokenizer, input_ids, method
            )

        elif method == "top_k":
            out_json, full_results_list = get_outputs(
                model, tokenizer, input_ids, method, top_k=5
            )

        elif method == "top_p":
            out_json, full_results_list = get_outputs(
                model, tokenizer, input_ids, method, top_p=0.80
            )

        elif method == "topp_topk":
            out_json, full_results_list = get_outputs(
                model, tokenizer, input_ids, method, top_k=50, top_p=0.80
            )

        outputs.append(full_results_list)

    # Get the outputs
    outputs = [item for sublist in outputs for item in sublist]

    outputs = pd.DataFrame(outputs).set_index("algorithm_name")
    outputs = outputs[sorted(outputs.columns)]

    # Write the outputs to an excel file so we can put it in
    # google sheets
    with pd.ExcelWriter("./results/task_1.xlsx") as writer:
        outputs.to_excel(writer, sheet_name="Task 1")

    # Code for Task 2
    tokenizer = T5Tokenizer.from_pretrained("hetpandya/t5-base-tapaco")
    model = T5ForConditionalGeneration.from_pretrained("hetpandya/t5-base-tapaco")

    # Load the dataset
    dataset = load_dataset("tapaco", "en", split="train")

    cnt = 0
    response_data = []
    unique_sentences = np.unique(dataset["paraphrase_set_id"])
    for uid in unique_sentences:
        filt_dataset = dataset.filter(
            lambda example: example["paraphrase_set_id"] == uid
        )
        sentences = filt_dataset["paraphrase"]

        paraphrases = []
        paraphrases.append(sentences[0])
        paraphrases.append(sentences[1])

        print(sentences)

        for method in methods:
            paraphrase = get_paraphrase(sentences[0], model, tokenizer, strategy=method)
            paraphrases.append(paraphrase)

        response_data.append(paraphrases)
        cnt += 1

        if cnt == 50:
            break

    response_df = pd.DataFrame(
        response_data,
        columns=[
            "original",
            "paraphrase",
            "greedy",
            "beam",
            "sampling",
            "top_k",
            "top_p",
            "topp_topk",
        ],
    )

    # Task 3 create metrics
    metrics = {
        "bleu": "bleu",
        "rouge": "rouge1",
        "meteor": "meteor",
    }

    # Iterate over the different metrics
    results_dict = {}

    # Get all of the reference texts
    references = response_df["original"].values
    for index, reference in tqdm(enumerate(references)):
        references = [[reference]]

        scores = {}
        for metric_name, metric_value in metrics.items():
            metric = evaluate.load(metric_name)
            bert_score = evaluate.load("bertscore")

            # Calcualte the metric for each method
            methods = ["greedy", "beam", "sampling", "top_k", "top_p", "topp_topk"]

            # Iterate over the methods and get the metrics
            for method in methods:
                prediction_string = response_df.iloc[index, :][method]
                predictions = [prediction_string]
                metric_results = metric.compute(
                    predictions=predictions, references=references
                )[metric_value]
                bert_score_results = bert_score.compute(
                    predictions=predictions,
                    references=references,
                    model_type="bert-base-uncased",
                )["f1"][0]

                scores[method + "_" + metric_name] = metric_results
                scores[method + "_bert_score_f1"] = bert_score_results

        # Get the results stored
        results_dict[reference] = scores

    results_and_metrics_df = pd.DataFrame(results_dict).T

    with pd.ExcelWriter("./results/responses_and_metrics.xlsx") as writer:
        response_df.to_excel(writer, sheet_name="Reponses")
        results_and_metrics_df.to_excel(writer, sheet_name="Metrics")


if __name__ == "__main__":
    run_tasks()
