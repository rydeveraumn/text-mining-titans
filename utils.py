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
