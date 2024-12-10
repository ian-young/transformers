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

from datasets import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def generate_squad_format(chunks, qa_model, look_back=1, look_ahead=1):
    """
    Generates a SQuAD-like QA dataset from document chunks.

    Args:
        chunks (list): List of document text chunks.
        qa_model (PreTrainedModel): Pre-trained QA model for generating questions/answers.
        look_back (int): Number of chunks to include before the current one as context.
        look_ahead (int): Number of chunks to include after the current one as context.

    Returns:
        list: List of dictionaries in SQuAD format.
    """

    squad_data = []

    for i, chunk in enumerate(chunks):
        # Context with look-back/look-ahead chunks
        context = (
            "".join(chunks[max(0, i - look_back) : i])
            + chunk
            + "".join(chunks[i + 1 : i + 1 + look_ahead])
        )

        # Generate questions and answers
        qa_prompt = f"Generate questions and answers for: {context}"
        qa_outputs = qa_model(
            qa_prompt, max_length=512, num_return_sequences=5
        )

        for output in qa_outputs:
            # Validate and parse generated text
            generated_text = output["generated_text"]
            if "? Answer:" in generated_text:
                question, answer = generated_text.split(
                    "? Answer: ", maxsplit=1
                )
                squad_data.append(
                    {
                        "context": context,
                        "question": f"{question.strip()}?",
                        "answer": answer.strip(),
                    }
                )

    return squad_data


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


def preprocess_custom_data(file_name, tokenizer, qa_model):
    """
    Preprocesses custom text data for question-answering model training.

    This function loads text data from a file, splits it into documents,
    and prepares it for training a QA model.

    Args:
        file_name (str): Path to the text file containing the dataset.
        tokenizer: Tokenizer used to process text into tokens.
        qa_model: Question-answering model used for generating training
            data.

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
    squad_data = generate_squad_format(chunked_docs, qa_model, tokenizer)
    train_data, val_data = train_test_split(
        squad_data, test_size=0.2, random_state=42
    )
    train_dataset = prepare_squad_dataset(train_data)
    val_dataset = prepare_squad_dataset(val_data)
    return {"train": train_dataset, "validation": val_dataset}, chunked_docs
