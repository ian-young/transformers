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
from multiprocessing import Pool, cpu_count
from re import match, split

from ollama import chat, ChatResponse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CHECKPOINT_FILE = "squad_data.jsonl"

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


def replace_unicode(text):
    """
    Replaces unicode characters in a string with their ASCII equivalents.

    This function takes a string as input and replaces specific unicode
    characters, such as curly quotes and apostrophes, with their standard
    ASCII counterparts.

    Args:
        text (str): The input string containing unicode characters.

    Returns:
        str: The string with unicode characters replaced by ASCII
            equivalents.

    Examples:
        replaced_text = replace_unicode("This string has curly quotes \u201clike this\u201d.")
    """
    unicode_replacements = {"\u2019": "'", "\u201c": '"', "\u201d": '"'}
    for unicode_char, replacement in unicode_replacements.items():
        text = text.replace(unicode_char, replacement)
    return text


def process_chunks(
    processed_indices,
    chunks,
    look_back,
    batch_size,
    checkpoint_file=CHECKPOINT_FILE,
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
        look_back (int): The number of previous chunks to include as
            context.
        batch_size (int): The number of entries to accumulate before
            writing to the checkpoint file.
        checkpoint_file (str): The path to the file where the processed
            data will be saved. (Optional)
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

        question_response: ChatResponse = chat(
            model="mistral:instruct",
            messages=[
                {
                    "role": "system",
                    "content": "Create a question as though you are a "
                    "user based on the given data. The question "
                    "should be clear, friendly, precise, and "
                    "concise. It should be no longer than a few "
                    "sentences.",
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
        )
        answer_response: ChatResponse = chat(
            model="mistral:instruct",
            messages=[
                {
                    "role": "system",
                    "content": "Create a consultative answer based"
                    "on the given data and question. The answer "
                    "should be clear,friendly, precise, and "
                    "concise. It should be no longer than a few "
                    "sentences.",
                },
                {
                    "role": "user",
                    "content": f"{question_response['message']['content']} {context}",
                },
            ],
        )

        question = question_response["message"]["content"]
        answer = answer_response["message"]["content"]
        # Write to file in MLX format. See:
        # https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data
        mlx_entry = {
            "prompt": replace_unicode(question),
            "completion": replace_unicode(answer),
        }
        batch_entries.append(mlx_entry)

        if len(batch_entries) >= batch_size:
            with open(checkpoint_file, "a", encoding="utf-8") as file:
                for entry in batch_entries:
                    file.write(json.dumps(entry) + "\n")
            batch_entries = []  # Reset the batch

            progress_bar.update(1)

    progress_bar.write("Writing remaining entries to checkpoint.")
    progress_bar.close()
    if batch_entries:
        with open(checkpoint_file, "a", encoding="utf-8") as file:
            for entry in batch_entries:
                file.write(json.dumps(entry) + "\n")


def generate_squad_format_with_checkpoint(
    chunks,
    look_back=1,
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

    if not processed_indices:
        process_chunks(
            processed_indices,
            chunks,
            look_back,
            batch_size,
        )
    elif max(processed_indices) != len(chunks) - 1:
        print(
            f"Continuing from {max(processed_indices)} to get to {len(chunks) - 1}."
        )
        process_chunks(
            processed_indices,
            chunks,
            look_back,
            batch_size,
        )
    else:
        print("Chunks already processed. Skipping...")


# Function to split the text into chunks of max_length
def chunk_text(data, overlap=50):
    """
    Splits a given text into smaller chunks based on a maximum token size.

    This function encodes the input text into tokens and divides it into
    manageable chunks that do not exceed the specified maximum size. It ensures
    that each chunk is properly decoded back into text format, making it
    suitable for further processing.

    Args:
        text (dict): The dictionary with input text to be chunked.
        overlap (int, optional): The amount of allowed overlap chunks.

    Returns:
        list: A list of dictionaries with text chunks, each containing up to max_size tokens.

    Examples:
        chunks = chunk_text("Your long text here...", max_size=192)
    """

    def is_number_with_decimal(sentence):
        # Check if the sentence is a number with decimal
        return bool(match(r"^\d+\.\d+$", sentence))

    sentences = split(r"(?<=[.!?]) +", data["text"])

    chunks = []
    current_chunk = []

    for sentence in sentences:
        if is_number_with_decimal(sentence):
            # Combine with the previous sentence if it's a number with decimal
            if current_chunk:
                current_chunk[-1] += f" {sentence}"
            else:
                current_chunk.append(sentence)
        else:
            current_chunk.append(sentence)

        if len(current_chunk) == 5 or sentence.endswith("."):
            # Join the current chunk into a single string
            chunk = " ".join(current_chunk).strip()

            # Add metadata to the chunk
            chunks.append(
                {
                    "url": data["url"],
                    "data": chunk,
                }
            )

            # Maintain overlap by keeping the last `overlap` sentences for the next chunk
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
    return chunks


def preprocess_custom_data(file_name, generate_squad):
    """
    Preprocesses custom text data for question-answering model training.

    This function loads text data from a file, splits it into documents,
    and prepares it for training a QA model.

    Args:
        file_name (str): Path to the text file containing the dataset.
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
        chunked_docs = pool.starmap(chunk_text, documents)

    chunked_docs = [item for sublist in chunked_docs for item in sublist]
    if generate_squad:
        # Generate SQuAD-like data
        generate_squad_format_with_checkpoint(
            chunks=chunked_docs,
        )

    print("Splitting training and testing data")
    squad_data = []
    with open("squad_data.jsonl", "r", encoding="UTF-8") as file:
        squad_data.extend(json.loads(line.strip()) for line in file)
    train_data, test_data = train_test_split(squad_data, test_size=0.2)
    test_data, validation_data = train_test_split(test_data, test_size=1 / 3)

    with open("data/train.jsonl", "w", encoding="utf-8") as file:
        file.write(train_data)

    with open("data/test.jsonl", "w", encoding="utf-8") as file:
        file.write(test_data)

    with open("data/valid.jsonl", "w", encoding="utf-8") as file:
        file.write(validation_data)
