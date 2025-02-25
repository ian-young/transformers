"""
Author: Ian Young

This module provides functionality for training a GPT-2 model using scraped
text data.

It includes functions for scraping data from websites, processing the text
into manageable chunks, tokenizing the data, and fine-tuning the GPT-2 model.
The module also defines a callback to pause training at specified intervals
to manage resource usage.

Classes:
    PauseTrainingCallback: A callback to pause training after a specified
    number of steps.

Functions:
    scrape_and_save: Scrapes data from predefined websites and saves it to
        a file.
    chunk_text: Splits a given text into smaller chunks based on a maximum
        token size.
    tokenize_function: Tokenizes input text examples and prepares them for
        model training.
    train_model: Trains a model using text data from a specified file.

Usage:
    The module can be imported and used to scrape data, train a model, and
    fine-tune it for specific tasks. It is designed to handle the entire
    workflow from data collection to model training.
"""

# pylint: disable=redefined-outer-name
import subprocess
from threading import Lock

import app.scrape as scrape  # Importing the scrape functionality
import numpy as np
import evaluate
import torch
from app.preprocess_data import preprocess_custom_data


def set_torch_device(model=None):
    """Sets the appropriate device for the model based on the available
    hardware.

    Args:
        model (optional): The model to be set on the appropriate device.

    Returns:
        model: The model with the device set.
        device (str): The device being used (GPU, CPU, or MPS).
    """
    # Check if MPS is available on Apple Silicon
    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"

    if model:
        model.to(device_name)

    return model, device_name


def scrape_and_save():
    """
    Scrapes data from predefined websites and saves it to a file.

    This function initiates the web scraping process by visiting a list of
    specified URLs. It collects data from these sites and ensures that the
    scraping is thread-safe by using a lock mechanism.

    Args:
        None

    Returns:
        None

    Examples:
        scrape_and_save()
    """
    print("Scraping websites to gather data...")
    visited_urls = set()
    lock = Lock()
    urls = [
        "https://help.verkada.com",  # Main help site
        "https://docs.verkada.com",  # Product docs
        "https://apidocs.verkada.com",  # API docs
        "https://verkada.com",  # Home page
        "https://verkada.com/pricing",  # Pricing page
    ]
    scrape.scrape_website(
        urls, visited_urls, lock
    )  # Calling the scrape function to save data
    print("Scraping complete, data saved to verkada_data.txt.")


def compute_metrics(p, tokenizer):
    """
    Computes the Exact Match (EM) and F1 score for the model predictions.

    Args:
        p: Tuple containing the predictions and labels
            p.predictions (numpy array): Model's predicted token IDs
            p.label_ids (numpy array): Ground truth token IDs
        tokenizer (PreTrainedTokenizer): The tokenizer associated with
            the model.

    Returns:
        dict: Dictionary containing the EM and F1 scores
    """
    # Load the ROUGE metric
    rouge = evaluate.load("rouge")

    # Get predictions and labels
    logits = p.predictions[0]
    pred_token_ids = np.argmax(logits, axis=-1)
    labels = p.label_ids

    # Decode the predicted and label token IDs to text
    decoded_preds = tokenizer.batch_decode(
        pred_token_ids, skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip leading/trailing whitespaces from the predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute ROUGE score
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)


def train_model(file_name, generate_squad):
    """
    Trains a model using text data from a specified file.

    This function loads text data, processes it into manageable chunks,
    and fine-tunes the provided model using the processed dataset. It
    sets up training parameters, initializes a trainer, and saves the
    fine-tuned model and tokenizer after training.

    Args:
        file_name (str): The path to the file containing the training data.
        generate_squad (bool): Tells the model whether or not it needs to
            generate new SQuAD data.

    Returns:
        PreTrainedModel: The custom-trained transformer.

    Examples:
        train_model(my_model, my_tokenizer, "data.txt")
    """
    command = [
        "mlx_lm.lora",
        "--train",
        "--model",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "--data",
        "data/",
        "--batch-size",
        "2",
    ]
    preprocess_custom_data(
        file_name,
        generate_squad,
    )

    print("Starting fine-tuning process.")
    try:
        # Start the process and open stdout in text mode
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Continuously read and print the output as it becomes available
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break  # No more output and the process has finished
            if output:
                print(output.strip())  # Print the output line by line

        # Wait for the command to complete and get the return code
        rc = process.poll()

        if rc != 0:
            raise subprocess.CalledProcessError(rc, command)

    except subprocess.CalledProcessError as e:
        # Handle errors that occurred during command execution
        print(f"Command failed with return code {e.returncode}")
        print(f"Output: {e.output}")
