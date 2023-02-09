def add_ending_index(answers, contexts):
    """
    Function for the question and asnwering task that
    also adds the ending index for the expected answers
    """
    # Loop through the answer and context pairs
    for answer, context in zip(answers, contexts):
        # Get the answer within the text
        answer_within_text = answer["text"]

        # We already know the starting index
        start_idx = answer["answer_start"]

        # And we can calculate the ending index
        end_idx = start_idx + len(answer_within_text)

        # We need to handle the case when this is not always
        # true
        if context[start_idx:end_idx] == answer_within_text:
            answer["answer_end"] = end_idx

        else:
            for n in [1, 2]:
                if context[(start_idx - n) : (end_idx - n)] == answer_within_text:
                    answer["answer_start"] = start_idx - n
                    answer["answer_end"] = end_idx - n


def trainer(data_loader, model, optimizer, epochs):
    """
    Function that runs a pytorch based training. For the model training
    with question and answering we will need the input_ids,
    attention mask, start and ending positions
    """
    # TODO: Set up loss meter
    # TODO: Get working with GPU

    # Put model in training mode
    model.train()

    for data in data_loader:
        # Zero out the gradients from the optimizer
        optimizer.zero_grad()

        # Get all of the outputs
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        start_positions = data["start_positions"]
        end_positions = data["end_positions"]

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

        # Calculate the gradients in the backward pass
        loss.backward()

        # update the gradients with the optimizer
        optimizer.step()
