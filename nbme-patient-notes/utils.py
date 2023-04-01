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
    max_length = 512  # Maximum sequence length
    apex = True  # Turn on mixed precision training


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
    for index, (inputs, label) in enumerate(train_loader):
        # zero out the gradients
        optimizer.zero_grad()

        # Get predictions
        with torch.cuda.amp.autocast(enabled=config.apex):
            predictions = model(inputs)

        # Get the loss
        loss = criterion(predictions.view(-1, 1), labels.view(-1, 1))

        # Mask the loss
        loss_mask = labels.view(-1, 1) != 1
        loss = torch.masked_select(loss, loss_mask).mean()

        # Gradient scaler
        scaler.scale(loss).backward()

        # Get the average loss
        losses += loss.item()

    return losses / len(train_loader)
