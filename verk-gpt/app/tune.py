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
from app.preprocess_data import preprocess_custom_data


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
