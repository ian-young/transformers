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

import logging
import json
import uuid
from multiprocessing import Pool, cpu_count
from os.path import exists
from re import sub

from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CHECKPOINT_FILE = "squad_data.jsonl"

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


def process_chunks(
    processed_indices,
    chunks,
    qa_model,
    tokenizer,
    device_name,
    look_back,
    checkpoint_file,
    batch_size,
):
    """Processes text chunks to generate question-answer pairs.

    This function iterates through text chunks, generates questions
    and answers using a question-answering model, and saves the results
    in SQuAD format. It includes logic to handle context, tokenization,
    answer extraction, and batch processing.


    Args:
        processed_indices (list): A list of indices of already processed
            chunks.
        chunks (list): A list of text chunks, where each chunk is a
            dictionary with 'data' (text content) and 'url' keys.
        qa_model (transformers.Pipeline): The question-answering pipeline.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for
            the model.
        device_name (str): The device to use for processing
            (e.g., 'cuda', 'cpu').
        look_back (int): The number of previous chunks to include as
            context.
        checkpoint_file (str): The path to the file where the processed
            data will be saved.
        batch_size (int): The number of entries to accumulate before
            writing to the checkpoint file.
    """
    batch_entries = []

    progress_bar = tqdm(
        total=len(chunks),
        desc="Processing chunks",
        unit="chunk",
        colour="cyan",
        dynamic_ncols=True,
        smoothing=0.5,
    )
    for i, chunk in enumerate(chunks):
        if i in processed_indices or i < len(processed_indices):
            progress_bar.update(1)
            continue  # Skip already processed indices

        # Context with look-back
        context = (
            "".join(
                [chunk["data"] for chunk in chunks[max(0, i - look_back) : i]]
            )
            + chunk["data"]
        )

        # Tokenize input
        inputs = tokenizer(context, return_tensors="pt", truncation=True).to(
            device_name
        )

        try:
            # Generate question-answer pair using the pipeline
            outputs = qa_model.generate(
                **inputs, max_length=100
            )  # Generate with the model directly
            qa_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

            qa_text = qa_text.replace(tokenizer.pad_token, "").replace(
                tokenizer.eos_token, ""
            )
            # Remove padding tokens at the start of questions
            qa_text = sub(r"<pad>\s*", "", qa_text)

            # Split into question and answer
            if tokenizer.sep_token in qa_text:

                question, answer = qa_text.split(
                    tokenizer.sep_token, maxsplit=1
                )
                normalized_answer = sub("/s+", "", answer.strip())

                answer_start = context.find(answer.strip())
                if answer_start == -1:
                    progress_bar.write(
                        f"Skipping chunk {i} | Answer not found in context."
                    )
                    progress_bar.update(1)
                    continue

                if normalized_answer in ["Verkada", "Verkada Inc."]:
                    progress_bar.write(
                        f"Skipping chunk {i} | Answer is too simple."
                    )
                    progress_bar.update(1)
                    continue

                # Write to file in SQuAD format. See:
                # https://huggingface.co/datasets/rajpurkar/squad
                squad_entry = {
                    "answers": {
                        "answer_start": answer_start,
                        "text": normalized_answer,
                    },
                    "context": context,
                    "id": str(uuid.uuid4()),  # Unique ID
                    "question": question.strip(),
                    "title": chunk["url"],  # Placeholder title
                    "index": i,
                }
                batch_entries.append(squad_entry)

                if len(batch_entries) >= batch_size:
                    with open(checkpoint_file, "a", encoding="utf-8") as file:
                        for entry in batch_entries:
                            file.write(json.dumps(entry) + "\n")
                    batch_entries = []  # Reset the batch

            else:
                progress_bar.write(
                    f"Skipping chunk {i} | Failed to generate valid Q&A format."
                )

            progress_bar.update(1)

        except (ValueError, KeyError, IndexError) as e:
            print(f"Error processing chunk {i} | {e}")

    progress_bar.write("Writing remaining entries to checkpoint.")
    progress_bar.close()
    if batch_entries:
        with open(checkpoint_file, "a", encoding="utf-8") as file:
            for entry in batch_entries:
                file.write(json.dumps(entry) + "\n")


def generate_squad_format_with_checkpoint(
    chunks,
    qa_model,
    tokenizer,
    device_name,
    look_back=1,
    checkpoint_file=CHECKPOINT_FILE,
    batch_size=75,
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
        print("Resuming from SQuAD checkpoint...")
        with open(checkpoint_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())
                processed_indices.add(data["index"])

    if not processed_indices:
        process_chunks(
            processed_indices,
            chunks,
            qa_model,
            tokenizer,
            device_name,
            look_back,
            checkpoint_file,
            batch_size,
        )
    elif max(processed_indices) != len(chunks) - 1:
        print(
            f"Continuing from {max(processed_indices)} to get to {len(chunks) - 1}."
        )
        process_chunks(
            processed_indices,
            chunks,
            qa_model,
            tokenizer,
            device_name,
            look_back,
            checkpoint_file,
            batch_size,
        )
    else:
        print("Chunks already processed. Skipping...")


def prepare_squad_dataset(squad_data, tokenizer):
    """
    Converts SQuAD data into a Hugging Face Dataset object.

    Args:
        squad_data (list): List of SQuAD-like dictionaries.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding.

    Returns:
        Dataset: Hugging Face Dataset.
    """
    tokenized_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for item in squad_data:
        question = item["question"]
        context = item["context"]
        answer = item["answers"]["text"]

        # Prepare the input
        input_text = f"question: {question} context: {context}"

        # Tokenize the question and context
        encoding = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",  # Or "longest", or "do_not_pad"
            max_length=192,
            return_tensors="pt",
        )

        # Tokenize the answer for labels
        target_encoding = tokenizer(
            answer,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        # Extract the tokenized input_ids and attention_mask
        input_ids = encoding["input_ids"].squeeze().tolist()
        attention_mask = encoding["attention_mask"].squeeze().tolist()
        labels = target_encoding["input_ids"].squeeze().tolist()

        # Append the tokenized values
        tokenized_data["input_ids"].append(input_ids)
        tokenized_data["attention_mask"].append(attention_mask)
        tokenized_data["labels"].append(labels)

    # Convert to Hugging Face Dataset
    return Dataset.from_dict(tokenized_data)


# Function to split the text into chunks of max_length
def chunk_text(data, tokenizer, max_size=192, overlap=50):
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
            chunk. Defaults to 192.
        overlap (int, optional): The amount of allowed overlap chunks.

    Returns:
        list: A list of dictionaries with text chunks, each containing up to max_size tokens.

    Examples:
        chunks = chunk_text("Your long text here...", max_size=192)
    """
    chunks = []

    # Encode the text
    tokens = tokenizer.encode(data["text"], add_special_tokens=False)

    # Trim if token list exceeds max_size
    while len(tokens) > max_size:
        part = tokens[:max_size]  # Grab a slice that fits the limit

        chunks.append(
            {
                "url": data["url"],
                "data": tokenizer.decode(part, skip_special_tokens=True),
            }
        )

        # Move forward by max_size - overlap
        tokens = tokens[max_size - overlap :]

    # Check for any remaining tokens
    if tokens:
        chunks.append(
            {
                "url": data["url"],
                "data": tokenizer.decode(
                    tokens, skip_special_tokens=True
                ),  # Decode remaining tokens
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
    documents = [
        json.loads(doc) for doc in text_data.split("\n\n") if doc.strip()
    ]
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
    squad_data = []
    with open("squad_data.jsonl", "r", encoding="UTF-8") as file:
        squad_data.extend(json.loads(line.strip()) for line in file)
    train_data, test_data = train_test_split(squad_data, test_size=0.2)
    test_data, validation_data = train_test_split(test_data, test_size=1 / 3)

    # Prepare Hugging Face datasets
    print("Preparing dataset")
    train_dataset = prepare_squad_dataset(train_data, tokenizer)
    test_dataset = prepare_squad_dataset(test_data, tokenizer)
    validation_dataset = prepare_squad_dataset(validation_data, tokenizer)

    return {
        "train": train_dataset,
        "test": test_dataset,
        "validation": validation_dataset,
    }, chunked_docs
