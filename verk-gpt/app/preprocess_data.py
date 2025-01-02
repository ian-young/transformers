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
import uuid
from multiprocessing import Pool, cpu_count
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
    checkpoint_file=CHECKPOINT_FILE,
    batch_size=100,
):
    """Generates SQuAD-formatted question-answer pairs from text chunks
    with checkpoint resuming capability.

    This function processes text chunks to generate question-answer pairs
    using a QA model. It supports resuming processing from a previous
    checkpoint and allows context windowing through look-back and look-ahead
    parameters.

    Args:
        chunks (list): A list of dictionaries with text chunks to process.
        qa_model (PreTrainedModel): The question-answering model used to
            generate Q&A pairs.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with
            the QA model.
        device_name (str): The device (e.g., 'cuda', 'cpu') to run the
            model on.
        look_back (int, optional): Number of chunks to include before the
            current chunk. Defaults to 1.
        checkpoint_file (str, optional): Path to the checkpoint file for
            resuming processing.
        batch_size (int, optional): How many batches to process in a chunk.

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

    batch_entries = []

    for i, chunk in tqdm(
        enumerate(chunks), total=len(chunks), desc="Processing chunks"
    ):
        if i in processed_indices:
            continue  # Skip already processed indices

        # Context with look-back
        context = "".join(chunks[max(0, i - look_back) : i]) + chunk["data"]

        # Tokenize input
        inputs = tokenizer(context, return_tensors="pt", truncation=True).to(
            device_name
        )

        try:
            # Generate question-answer pair using the pipeline
            outputs = qa_model.generate(
                **inputs, max_length=100
            )  # Generate with the model directly
            qa_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            qa_text = qa_text.replace(tokenizer.pad_token, "").replace(
                tokenizer.eos_token, ""
            )

            # Split into question and answer
            if tokenizer.sep_token in qa_text:
                question, answer = qa_text.split(
                    tokenizer.sep_token, maxsplit=1
                )

                answer_start = context.find(answer.strip())
                if answer_start == -1:
                    print(f"Skipping chunk {i}: Answer not found in context.")
                    continue

                # Write to file in SQuAD format
                squad_entry = {
                    "title": chunk["url"],  # Placeholder title
                    "paragraphs": [
                        {
                            "context": context,
                            "qas": [
                                {
                                    "id": str(uuid.uuid4()),  # Unique ID
                                    "question": question.strip(),
                                    "answers": [
                                        {
                                            "text": answer.strip(),
                                            "answer_start": answer_start,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    "index": i,
                }
                batch_entries.append(squad_entry)

                if len(batch_entries) >= batch_size:
                    with open(checkpoint_file, "a", encoding="utf-8") as file:
                        for entry in batch_entries:
                            file.write(json.dumps(entry) + "\n")
                    batch_entries = []  # Reset the batch

            else:
                print(
                    f"Skipping chunk {i}: Failed to generate valid Q&A format."
                )

        except (ValueError, KeyError, IndexError) as e:
            print(f"Error processing chunk {i}: {e}")

    if batch_entries:
        with open(checkpoint_file, "a", encoding="utf-8") as file:
            for entry in batch_entries:
                file.write(json.dumps(entry) + "\n")


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
def chunk_text(data, tokenizer, max_size=512, overlap=50):
    """
    Splits a given text into smaller chunks based on a maximum token size.

    This function encodes the input text into tokens and divides it into
    manageable chunks that do not exceed the specified maximum size. It ensures
    that each chunk is properly decoded back into text format, making it
    suitable for further processing.

    Args:
        text (dict): The dictionary with input text to be chunked.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with
            the model.
        max_size (int, optional): The maximum number of tokens allowed in each
            chunk. Defaults to 1024.
        overlap (int, optional): The amount of allowed overlap chunks.

    Returns:
        list: A list of dictionaries with text chunks, each containing up to max_size tokens.

    Examples:
        chunks = chunk_text("Your long text here...", max_size=512)
    """
    tokens = tokenizer.encode(data["text"], add_special_tokens=True)
    chunks = []

    # Handle chunking with overlap
    for i in range(0, len(tokens), max_size - overlap):
        part = tokens[i : i + max_size]
        chunks.append(
            {
                "url": data["url"],
                "data": tokenizer.decode(part, skip_special_token=True),
            }
        )

    return chunks


def preprocess_custom_data(
    file_name, tokenizer, qa_model, device_name, generate_squad
):
    """
    Preprocesses custom text data for question-answering model training.

    This function loads text data from a file, splits it into documents,
    and prepares it for training a QA model.

    Args:
        file_name (str): Path to the text file containing the dataset.
        tokenizer: Tokenizer used to process text into tokens.
        qa_model (PreTrainedModel): Question-answering model used for
            generating training data.
        device_name (str): The name of the primary compute device.
        generate_squad (bool): Tells the model whether or not it needs to
            generate new SQuAD data.

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

    with Pool(cpu_count()) as pool:
        chunked_docs = pool.starmap(
            chunk_text, [(doc, tokenizer) for doc in documents]
        )

    chunked_docs = [item for sublist in chunked_docs for item in sublist]

    if generate_squad:
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

    train_data, test_data = train_test_split(squad_data, test_size=0.2)
    test_data, validation_data = train_test_split(test_data, test_size=1 / 3)

    # Prepare Hugging Face datasets
    print("Preparing dataset")
    train_dataset = prepare_squad_dataset(train_data)
    test_dataset = prepare_squad_dataset(test_data)
    validation_dataset = prepare_squad_dataset(validation_data)

    return {
        "train": train_dataset,
        "test": test_dataset,
        "validation": validation_dataset,
    }, chunked_docs
