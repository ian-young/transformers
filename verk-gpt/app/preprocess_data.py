"""
Author: Ian Young

This module provides functionality for loading, preparing, and tokenizing
datasets for training machine learning models.

It includes functions to load datasets, preprocess examples based on their
type, split text into manageable chunks, and tokenize the text for use with
models like GPT-2. The module is designed to facilitate the preparation of
data for training and evaluation in natural language processing tasks.

Functions:
    load_and_prepare_dataset: Loads and prepares a dataset for training by
        preprocessing its examples.
    chunk_text: Splits a given text into smaller chunks based on a maximum
        token size.
    tokenize_function: Tokenizes input text examples and prepares them for
        model training.

Usage:
    This module can be imported and used to handle dataset loading and
    preprocessing tasks, making it easier to prepare data for training
    machine learning models.
"""

from datasets import load_dataset


def load_and_prepare_dataset(dataset_name, tokenizer):
    """
    Loads and prepares a dataset for training by preprocessing its examples.

    This function retrieves a specified dataset and applies preprocessing
    steps based on the dataset type. It formats the examples into a
    suitable structure for training, tokenizes the text, and sets the
    dataset format for use with PyTorch.

    Args:
        dataset_name (list): The list of names of the datasets to load and prepare.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with
            the model.

    Returns:
        tuple: A tuple containing the training dataset and the validation
            dataset (if available).

    Raises:
        ValueError: If the dataset name is unknown or unsupported.

    Examples:
        train_dataset, validation_dataset = load_and_prepare_dataset("squad")
    """
    dataset = load_dataset(dataset_name)

    def preprocess(example):
        if dataset_name == "roskoN/dailydialog":
            return {"text": " ".join(example["utterances"])}
        elif dataset_name == "AlekseyKorshuk/persona-chat":
            if "history" in example:
                utterances = [
                    utter["text"] if isinstance(utter, dict) else str(utter)
                    for utter in example["history"]
                ]
            else:
                # If 'history' is missing, you can handle this case differently,
                # for example, by skipping this example or processing differently.
                utterances = []

            candidates = example.get(
                "candidates", []
            )  # Safely get 'candidates' key

            # Return combined text and candidates
            return {"text": " ".join(utterances), "candidates": candidates}
        elif dataset_name in ["rajpurkar/squad", "rajpurkar/squad_v2"]:
            return {
                "text": f"Question: {example['question']} Answer: {example['answers']['text'][0]}"
            }
        elif dataset_name == "pfb30/multi_woz_v22":
            dialogues = []
            for i in range(len(example['turns']['turn_id'])):
                dialogue = ""
                for j in range(i, len(example['turns']['turn_id'])):
                    speaker = example['turns']['speaker'][j]
                    utterance = example['turns']['utterance'][j]
                    dialogue += f"[Speaker {speaker}] {utterance}\n"
                dialogues.append(dialogue.strip())
            # Return dialogues as a dictionary with 'text' as the key
            # Join the dialogues into a single string to match tokenizer's expected input
            return {"text": " ".join(dialogues)}
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = dataset.map(
        preprocess, remove_columns=dataset.column_names["train"]
    )
    dataset = dataset.map(
        lambda examples: tokenize_function(
            examples=examples, tokenizer=tokenizer
        ),
        batched=True,
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return dataset["train"], (
        dataset["validation"] if "validation" in dataset else None
    )


# Function to split the text into chunks of max_length
def chunk_text(text, tokenizer, max_size=1024):
    """
    Splits a given text into smaller chunks based on a maximum token size.

    This function encodes the input text into tokens and divides it into
    manageable chunks that do not exceed the specified maximum size. It ensures
    that each chunk is properly decoded back into text format, making it
    suitable for further processing.

    Args:
        text (str): The input text to be chunked.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with
            the model.
        max_size (int, optional): The maximum number of tokens allowed in each
            chunk. Defaults to 1024.

    Returns:
        list: A list of text chunks, each containing up to max_size tokens.

    Examples:
        chunks = chunk_text("Your long text here...", max_size=512)
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = []

    # If the token list exceeds max_size, split further by tokens
    while len(tokens) > max_size:
        # print(f"Chunk exceeds max size of {max_size}, splitting further...")
        part = tokens[:max_size]  # Get a part that fits within the token limit
        chunks.append(tokenizer.decode(part, skip_special_tokens=True))
        tokens = tokens[max_size:]  # Move to the next part

    # After splitting, check for any remaining tokens and add them
    if len(tokens) > 0:
        chunks.append(tokenizer.decode(tokens, skip_special_tokens=True))

    return chunks


# Tokenize the dataset (GPT-2 requires text to be tokenized)
def tokenize_function(examples, tokenizer):
    """
    Tokenizes input text examples and prepares them for model training.

    This function takes a dictionary of examples, tokenizes the text, and
    ensures that the resulting token sequences are properly padded and
    truncated to a maximum length. It also creates labels that are identical
    to the input token IDs, making it suitable for supervised learning tasks.

    Args:
        examples (dict): A dictionary containing the text examples to be tokenized.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with
            the model.

    Returns:
        dict: A dictionary containing the tokenized inputs and labels.

    Examples:
        tokenized_inputs = tokenize_function(
            {"text": "Sample text for tokenization."}
        )
    """
    inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",  # Return as pytorch tensors
    )

    # Labels are the same as input_ids
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs
