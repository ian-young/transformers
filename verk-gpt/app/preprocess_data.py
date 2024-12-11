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

import json
from os.path import exists

from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CHECKPOINT_FILE = "squad_data.json"


def generate_squad_format_with_checkpoint(
    chunks,
    qa_model,
    tokenizer,
    device_name,
    look_back=1,
    look_ahead=1,
    checkpoint_file=CHECKPOINT_FILE,
):
    """Generates SQuAD-formatted question-answer pairs from text chunks
    with checkpoint resuming capability.

    This function processes text chunks to generate question-answer pairs
    using a QA model. It supports resuming processing from a previous
    checkpoint and allows context windowing through look-back and look-ahead
    parameters.

    Args:
        chunks (list): A list of text chunks to process.
        qa_model (PreTrainedModel): The question-answering model used to
            generate Q&A pairs.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with
            the QA model.
        device_name (str): The device (e.g., 'cuda', 'cpu') to run the
            model on.
        look_back (int, optional): Number of chunks to include before the
            current chunk. Defaults to 1.
        look_ahead (int, optional): Number of chunks to include after the
            current chunk. Defaults to 1.
        checkpoint_file (str, optional): Path to the checkpoint file for
            resuming processing.

    Returns:
        None: Writes SQuAD-formatted Q&A pairs to the checkpoint file.

    Raises:
        ValueError: If there are issues processing a chunk.
        KeyError: If there are missing keys in the model output.
        IndexError: If there are indexing issues during processing.
    """
    processed_indices = set()

    # Load previously processed indices to resume from the last checkpoint
    if exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())
                processed_indices.add(data["index"])

    with open(checkpoint_file, "a", encoding="utf-8") as file:
        for i, chunk in tqdm(
            enumerate(chunks), total=len(chunks), desc="Processing chunks"
        ):
            if i in processed_indices:
                continue  # Skip already processed indices

            # Context with look-back and look-ahead
            context = (
                "".join(chunks[max(0, i - look_back) : i])
                + chunk
                + "".join(chunks[i + 1 : i + 1 + look_ahead])
            )

            # Tokenize input
            inputs = tokenizer(context, return_tensors="pt").to(device_name)

            try:
                # Generate question-answer pair using the pipeline
                outputs = qa_model.generate(
                    **inputs, max_length=100
                )  # Generate with the model directly
                qa_text = tokenizer.decode(
                    outputs[0], skip_special_tokens=False
                )

                qa_text = qa_text.replace(tokenizer.pad_token, "").replace(
                    tokenizer.eos_token, ""
                )

                # Split into question and answer
                if tokenizer.sep_token in qa_text:
                    question, answer = qa_text.split(
                        tokenizer.sep_token, maxsplit=1
                    )

                    # Write to file in SQuAD format
                    squad_entry = {
                        "index": i,
                        "context": context,
                        "question": question.strip(),
                        "answer": answer.strip(),
                    }
                    file.write(json.dumps(squad_entry) + "\n")
                else:
                    print(
                        f"Skipping chunk {i}: Failed to generate valid Q&A format."
                    )
            except ValueError as e:
                print(f"ValueError processing chunk {i}: {e}")
            except KeyError as e:
                print(
                    f"KeyError processing chunk {i}: Missing key in output - {e}"
                )
            except IndexError as e:
                print(
                    f"IndexError processing chunk {i}: Index out of range - {e}"
                )


def prepare_squad_dataset(squad_data):
    """
    Converts SQuAD data into a Hugging Face Dataset object.

    Args:
        squad_data (list): List of SQuAD-like dictionaries.

    Returns:
        Dataset: Hugging Face Dataset.
    """
    return Dataset.from_list(squad_data)


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


def preprocess_custom_data(file_name, tokenizer, qa_model, device_name):
    """
    Preprocesses custom text data for question-answering model training.

    This function loads text data from a file, splits it into documents,
    and prepares it for training a QA model.

    Args:
        file_name (str): Path to the text file containing the dataset.
        tokenizer: Tokenizer used to process text into tokens.
        qa_model (PreTrainedModel): Question-answering model used for generating training
            data.
        device_name: The name of the primary compute device

    Returns:
        A processed dataset ready for model training.

    Examples:
        dataset = preprocess_custom_data('data.txt', tokenizer, qa_model)
    """
    # Load the scraped data
    with open(file_name, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Split and chunk the text
    documents = text_data.split("\n\n")
    chunked_docs = []
    print("Chunking large documents...")
    for doc in tqdm(
        documents, total=len(documents), desc="Processing documents"
    ):
        chunked_docs.extend(chunk_text(doc, tokenizer=tokenizer))

    # Generate SQuAD-like data
    generate_squad_format_with_checkpoint(
        chunks=chunked_docs,
        qa_model=qa_model,
        tokenizer=tokenizer,
        device_name=device_name,
    )

    print("Splitting training and testing data")
    squad_data = None
    with open("squad_data.txt", "r", encoding="UTF-8") as file:
        squad_data = file.read()

    train_data, val_data = train_test_split(
        squad_data, test_size=0.2, random_state=42
    )
    print("Preparing dataset")
    train_dataset = prepare_squad_dataset(train_data)
    val_dataset = prepare_squad_dataset(val_data)
    return {"train": train_dataset, "validation": val_dataset}, chunked_docs
