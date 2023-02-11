# stdlib
import collections

# third party
import evaluate
import numpy as np
import torch
import tqdm


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


def train_single_epoch(data_loader, model, optimizer, device):
    """
    Function that runs a pytorch based training. For the model training
    with question and answering we will need the input_ids,
    attention mask, start and ending positions
    """
    # TODO: Set up loss meter
    # TODO: Get working with GPU
    step_losses = []

    # Put model in training mode
    model.train()

    # Description of training
    tqdm_loop = tqdm.tqdm(data_loader, leave=True)

    for data in tqdm_loop:
        # Zero out the gradients from the optimizer
        optimizer.zero_grad()

        # Get all of the outputs
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        start_positions = data["start_positions"].to(device)
        end_positions = data["end_positions"].to(device)

        # Get the outputs of the model - remember that
        # at least with the QA models the first output of the
        # model will be the loss
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        # Gather the loss
        loss = outputs.loss
        step_losses.append(loss)

        # Calculate the gradients in the backward pass
        loss.backward()

        # update the gradients with the optimizer
        optimizer.step()

        # tqdm description
        tqdm_loop.set_postfix(loss=loss.item())

    return step_losses


def question_and_answer_evaluation(
    model, data_loader, validation_datasets, device
):  # noqa
    """
    Function to perform the evaluation for the question and answer
    validation set.
    """
    start_logits_list = []
    end_logits_list = []

    # tqdm loop for validation
    tqdm_loop = tqdm.tqdm(data_loader, leave=True)

    # Get the predictions from the model
    model.eval()
    with torch.no_grad():
        for data in tqdm_loop:
            # Get the model inputs
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            # Get the predictions
            predictions = model(input_ids, attention_mask=attention_mask)
            start_logits = predictions.start_logits.detach().cpu().numpy()
            end_logits = predictions.end_logits.detach().cpu().numpy()

            # append to list to get the full prediction set
            start_logits_list.append(start_logits)
            end_logits_list.append(end_logits)

    # Concatenate the predictions
    start_logits = np.concatenate(start_logits_list)
    end_logits = np.concatenate(end_logits_list)

    # Get the tokenized dataset
    eval_set = validation_datasets["validation_data"]
    raw_validation_data = validation_datasets["raw_validation_data"]

    # Set up the example to features from the evalset
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(eval_set):
        example_to_features[feature["example_id"]].append(idx)

    # Set parameters for finding best answer
    n_best = 20
    max_answer_length = 30
    predicted_answers = []

    for example in raw_validation_data:
        # Get the example id from the original validation
        # dataset
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Iterate over the different features
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]

            # Get the start and end indexes
            # We need to resort to get the highest values
            start_indexes = np.argsort(start_logit)[::-1][:n_best]
            end_indexes = np.argsort(end_logit)[::-1][:n_best]

            # Iterate throught the indexes to get the answer
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip over answers that are not fully in the context
                    if (
                        offsets[start_index] is None
                        or offsets[end_index] is None  # noqa
                    ):  # noqa
                        continue

                    # Skip answers with a length that is
                    # either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answers.append(
                        {
                            "text": context[
                                offsets[start_index][0] : offsets[end_index][1]  # noqa
                            ],
                            "logit_score": start_logit[start_index]
                            + end_logit[end_index],
                        }
                    )

        # Get the best answer for evaluating the metric
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answers.append(
            {
                "id": example_id,
                "prediction_text": best_answer["text"],
                "no_answer_probability": 0.0,
            }
        )

    # Setup the metric
    metric = evaluate.load("squad_v2")

    # Set up the answers
    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in raw_validation_data  # noqa
    ]

    # Return the metric
    return metric.compute(
        predictions=predicted_answers, references=theoretical_answers
    )  # noqa
