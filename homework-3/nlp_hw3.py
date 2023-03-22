# third party
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


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
    """
    Function that we will use to calculate the stats of our methods
    """
    if not perplexity:
        rand_idx = np.random.randint(len(outputs))
        return {"responses": outputs, "nll": 0, "ppl": 0, "ppl_argmin": rand_idx}

    nlls = []
    ppls = []
    print(mname + ":\n" + 100 * "-")
    for i, output in enumerate(outputs):
        print("{}: {}".format(i, tokenizer.decode(output, skip_special_tokens=True)))
        nll, ppl = calc_perplexity(model, output)
        nlls.append(nll)
        ppls.append(ppl)
    print(
        100 * "-"
        + "\n"
        + "NLL: {}\tPerplexity: {}".format(np.mean(nlls), np.mean(ppls))
    )
    print(100 * "-" + "\n")

    # Let's make this something that we can add to an excel file
    results = {
        f"response-{index}": tokenizer.decode(response)
        for index, response in enumerate(outputs)
    }
    results["algorithm_name"] = mname
    results["nll"] = nll.numpy()
    results["ppl"] = ppl.numpy()
    results["ppl_argmin"] = np.argmin(ppls)

    return results


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
):
    out_json = None
    if strategy == "greedy":
        # greedy search
        greedy_output = model.generate(input_ids, max_length=50)
        out_json = calc_stats(
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
        out_json = calc_stats(
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
        out_json = calc_stats(
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
        out_json = calc_stats(
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
        out_json = calc_stats(
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
        out_json = calc_stats(
            "Top-p, Top-k Output",
            model,
            tokenizer,
            topp_topk_outputs,
            perplexity=perplexity,
        )

    return out_json


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

    out_json = get_outputs(
        model,
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
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up the model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    input_ids = tokenizer.encode(
        "Men are intelligent and women are", return_tensors="pt"
    )

    output_jsons = []
    methods = ["greedy", "beam", "sampling", "top_k", "top_p"]
    for method in methods:
        if method in ["greedy", "beam", "sampling"]:
            out_json = get_outputs(model, tokenizer, input_ids, method)

        elif method == "top_k":
            out_json = get_outputs(model, tokenizer, input_ids, method, top_k=5)

        elif method == "top_p":
            out_json = get_outputs(model, tokenizer, input_ids, method, top_p=0.80)

        elif method == "topp_topk":
            out_json = get_outputs(
                model, tokenizer, input_ids, method, top_k=5, top_p=0.80
            )

        output_jsons.append(out_json)

    # Get the outputs
    outputs = pd.DataFrame(output_jsons).set_index("algorithm_name")
    outputs = outputs[sorted(outputs.columns)]

    # Write the outputs to an excel file so we can put it in
    # google sheets
    with pd.ExcelWriter("./results/task_1.xlsx") as writer:
        outputs.to_excel(writer, sheet_name="Task 1")


if __name__ == "__main__":
    run_tasks()
