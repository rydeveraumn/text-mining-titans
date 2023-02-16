# stdlib
import logging
import os
import time
import timeit

# third party
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer
from transformers.data.metrics.squad_metrics import (
    compute_exact,
    compute_predictions_logits,
    normalize_answer,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult

# Create logger
logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        # The first thing we will do just like the training inner loop is get the
        # training dataloader
        start = time.time()
        train_loader = self.get_train_dataloader()
        eval_loader = self.get_eval_dataloader()

        # Get number of epochs and max steps
        number_of_epochs = args.num_train_epochs

        # In our case we will set our own optimizer internally
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)  # noqa

        # Implement a learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

        train_step_losses = []
        # Train the model over epochs
        for epoch in range(number_of_epochs):
            path = f"./model_weights/text-mining-titans-roberta-qa-cp{epoch}.pt"

            # Model is a part of the class noqa here
            self.model.train()  # noqa
            train_loss_per_epoch = 0

            with tqdm.tqdm(train_loader, unit="batch") as training_epoch:
                training_epoch.set_description(f"Training epoch {epoch}")
                for step, data in enumerate(train_loader):
                    # Zero out the optimizer
                    self.optimizer.zero_grad()

                    # Set up the inputs
                    input_ids = data["input_ids"].to(args.device)
                    attention_mask = data["attention_mask"].to(args.device)
                    start_positions = data["start_positions"].to(args.device)
                    end_positions = data["end_positions"].to(args.device)

                    # Get the outputs of the model
                    outputs = self.model(  # noqa
                        input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions,
                    )

                    # Get the loss
                    loss = outputs.loss

                    # Make the backward pass and compute the gradients
                    loss.backward()

                    # Apply gradients
                    self.optimizer.step()

                    # Set loss description
                    training_epoch.set_postfix(loss=loss.item())

                    # Update the training loss per epoch
                    train_loss_per_epoch += loss.item()
                    train_step_losses.append(loss.item())

            # Step with the scheduler after the epoch
            self.scheduler.step()
            train_loss_per_epoch /= len(train_loader)

            # Save the model
            torch.save(self.model.state_dict(), path)  # noqa

            # Setup the evaluation process at the end of each epoch
            eval_loss_per_epoch = 0
            self.model.eval()  # noqa

            with torch.no_grad():
                with tqdm.tqdm(eval_loader, unit="batch") as eval_epoch:
                    eval_epoch.set_description(f"Evaluation Epoch {epoch}")

                    for eval_step, eval_data in enumerate(eval_loader):
                        input_ids = eval_data["input_ids"].to(args.device)
                        attention_mask = eval_data["attention_mask"].to(args.device)
                        start_positions = eval_data["start_positions"].to(args.device)
                        end_positions = eval_data["end_positions"].to(args.device)

                        # Get the outputs of the model
                        outputs = self.model(  # noqa
                            input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions,
                        )

                        # Eval loss
                        eval_loss = outputs.loss
                        eval_loss_per_epoch += eval_loss.item()

                # compute the loss per epoch
                eval_loss_per_epoch /= len(eval_loader)

                # Print the training and evaluation losses
                print(f"Train Loss: {train_loss_per_epoch}")
                print(f"Eval Loss: {eval_loss_per_epoch}")

        # Get the end time
        end = time.time()
        print(f"Time: {(end - start) / 60.0}")

        # Save train step losses
        train_step_losses = pd.Series(train_step_losses)
        train_step_losses.to_csv("./results/train-step-losses.csv", index=False)


def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    """
    Function to prepare SQUAD v2 data for huggingface
    """
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the
    # overflows using a stride. This results
    # in one example possible giving several features when a
    # context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a
    # long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what
        # is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of
        # the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span
            # (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and
                # token_end_index to the two ends of the answer.
                # Note: we could go after the last
                # offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1
                )  # noqa
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_length, doc_stride):
    """
    Function to prepare SQUAD v2 data for huggingface
    """
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a
    # lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the
    # overflows using a stride. This results
    # in one example possible giving several features when a context is long,
    # each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized_examples.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(tokenized_examples["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = tokenized_examples.sequence_ids(i)
        offset = tokenized_examples["offset_mapping"][i]
        tokenized_examples["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    tokenized_examples["example_id"] = example_ids
    return tokenized_examples


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(output_dir, model, tokenizer, device, datasets, prefix=""):
    batch_size = 4
    model_type = "roberta"
    dataset, examples, features = datasets

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if model_type in [
                "xlm",
                "roberta",
                "distilbert",
                "camembert",
                "bart",
                "longformer",
            ]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # Get the predicted outputs
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs.to_tuple()]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info(
        "  Evaluation done in total %f secs (%f sec per example)",
        evalTime,
        evalTime / len(dataset),
    )

    # Compute predictions
    output_prediction_file = os.path.join(
        output_dir, "predictions_{}.json".format(prefix)
    )
    output_nbest_file = os.path.join(
        output_dir, "nbest_predictions_{}.json".format(prefix)
    )
    output_null_log_odds_file = os.path.join(
        output_dir, "null_odds_{}.json".format(prefix)
    )

    # TODO: Get defualt inputs for this function
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size=20,
        max_answer_length=30,
        do_lower_case=True,
        output_prediction_file=output_prediction_file,
        output_nbest_file=output_nbest_file,
        output_null_log_odds_file=output_null_log_odds_file,
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=0.0,
        tokenizer=tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results, examples, predictions


def build_incorrect_samples(examples, predictions):
    """
    Function that will define incorrect samples
    based on the exact match criteria
    """
    results = []

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [
            answer["text"]
            for answer in example.answers
            if normalize_answer(answer["text"])
        ]

        if not gold_answers:
            gold_answers = [""]

        if qas_id not in predictions:
            continue

        prediction = predictions[qas_id]
        exact_scores = [
            (
                compute_exact(a, prediction),
                a,
                prediction,
                example.context_text,
                example.question_text,
                qas_id,
            )
            for a in gold_answers
        ]
        exact_scores = sorted(exact_scores, key=lambda x: x[0], reverse=True)[0]
        results.append(exact_scores)

        df = pd.DataFrame(
            results,
            columns=[
                "exact_score",
                "answer",
                "prediction",
                "context",
                "question",
                "id",
            ],
        )
        df = df.loc[df["exact_score"] == 0].sample(frac=1.0).reset_index(drop=True)

        logging.info("Saving incorrect results!")
        df.head(10).to_csv("./results/incorrect-samples-v2.csv", index=False)
