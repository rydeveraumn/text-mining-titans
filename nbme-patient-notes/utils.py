# first party
import itertools

# third party
import torch
from transformers import AutoTokenizer


# set up the configuration class
class Configuration:
    """
    Project configuration class (commonly used in Kaggle)
    """

    # Splits for validation
    n_splits = 5

    # Model information
    model = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model)
    max_length = 354  # Maximum sequence length - comes from Kaggle notebook
    apex = True  # Turn on mixed precision training
    fc_dropout = 0.20


def training_function(
    config, train_loader, model, criterion, optimizer, scheduler, device
) -> float:
    """
    Function that will train a single epoch with pytorch functionality. This
    will utilize a criterion, optimizer, scheduler and device
    """
    # put model in training mode
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=config.apex)
    losses = 0.0

    # Iterate over data and train model
    for index, (inputs, labels) in tqdm.tqdm(enumerate(train_loader)):
        # zero out the gradients
        optimizer.zero_grad()

        # Put inputs and labels on device
        for k, v in inputs.items():
            inputs[k] = v.to(device=device)

        # labels on device
        labels = labels.to(device=device)

        # Get predictions
        with torch.cuda.amp.autocast(enabled=config.apex):
            predictions = model(inputs)

        # Get the loss
        loss = criterion(predictions.view(-1, 1), labels.view(-1, 1))

        # Mask the loss to not include outside tokens
        loss_mask = labels.view(-1, 1) != 1
        loss = torch.masked_select(loss, loss_mask).mean()

        # Gradient scaler
        scaler.scale(loss).backward()

        # Get the average loss
        losses += loss.item()

    return losses / len(train_loader)


def validation_function(config, valid_loader, model, device):
    """
    Function that will run validation after a single epoch
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for index, (inputs, labels) in enumerate(valid_loader):
            # Put inputs on device
            for k, v in inputs.items():
                inputs[k] = v.to(device=device)

            # labels on device
            labels = labels.to(device=device)

            # Get predictions
            predictions = model(inputs)
            predictions = torch.sigmoid(predictions.flatten())

            all_predictions.append(
                predictions.detach().numpy(), labels.flatten().detach().numpy()
            )

    # Concatenate the predictions and labels
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    return all_predictions, all_labels


#### Build Project Outputs ####


def get_character_probabilities(patient_notes, predictions, config):
    """
    For each of the patient notes get the probabilies related to the
    charater level
    """
    # Length on the original text and get length of the raw characters
    results = [np.zeros(len(patient_note)) for patient_note in patient_notes]

    # Get the predictions for the character level
    for index, (patient_note, prediction) in enumerate(zip(patient_notes, predictions)):
        # Get the encoding
        encoding = config.tokenizer(
            patient_note,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        # Get the offset mapping
        offset_mappings = encoding["offset_mapping"]
        for _, (offset_mapping, preds) in enumerate(zip(offset_mappings, prediction)):
            # Get starting and ending character index
            # The first and last offset mapping will have indexes
            # (0, 0)
            start, end = offset_mapping

            # Map the prediction to the character level
            results[index][start:end] = preds

    return results


def get_thresholded_sequences(character_probabilities, threshold=0.5):
    """
    Function that takes in the predicted probabilities, thresholds them
    and then returns a list of sequences
    """
    results = []
    for character_probabilitie in character_probabilities:
        # Threshold the predictions (add one because of the [CLS] type tokens)
        result = np.where(character_probabilitie >= threshold)[0] + 1

        # Line that creates the sequences
        result = [
            list(g)
            for _, g in itertools.groupby(
                result, key=lambda n, c=itertools.count(): n - next(c)
            )
        ]

        # Create the intervals for the sequences
        result = [f"{min(r)} {max(r)}" for r in result]

        # Join and append the results
        result = ";".join(result)
        results.append(result)

    return results


def get_predictions(results):
    """
    Function that gets the predicted output sequencees and creates
    predictions for scoring
    """
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(";")]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)

    return predictions


def create_labels_for_scoring(df):
    """
    Function to create labels for scoring;
    example: ['0 1', '3 4'] -> ['0 1; 3 4']
    """
    df["location_for_create_labels"] = [ast.literal_eval(f"[]")] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, "location"]
        if lst:
            new_lst = ";".join(lst)
            df.loc[i, "location_for_create_labels"] = ast.literal_eval(
                f'[["{new_lst}"]]'
            )
    # create labels
    truths = []
    for location_list in df["location_for_create_labels"].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(";")]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)

    return truths


#### Project Metric ####


def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Function to convert spans to a binary array indicating
    whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1

    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(
            np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0
        )
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))

    return micro_f1(bin_preds, bin_truths)
