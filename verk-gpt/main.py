"""
Author: Ian Young

This module serves as the main entry point for training and querying a
GPT-2 model.

It orchestrates the entire workflow, including scraping data from specified
websites, loading or initializing the model and tokenizer, training the
model with the scraped data, and testing the model with a predefined query.
The module also manages the logic for using a backup dataset and loading
from a checkpoint if available, while providing a graceful exit on keyboard
interruption.

Constants:
    USE_BACKUP (bool): Flag indicating whether to use a backup dataset.
    MODEL_NAME (str): The name of the base model to initialize if no
        checkpoint is found.
    QUERY (str): The custom query to test the fine-tuned model.
    CHECKPOINT_PATH (str): The path to the fine-tuned model checkpoint.

Functions:
    main: The main driver function for the GPT-2 model.

Usage:
    This module can be executed as a standalone script to initiate the
    training and querying process. It will handle all necessary steps
    from data scraping to model evaluation.
"""

from os.path import exists

import app
import app.chat_train as chat_train
from app.tune import (  # Importing the function to start training
    scrape_and_save,
    train_model,
)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

USE_BACKUP = True
TRAIN_CHAT = True
V_TRAIN = False
MODEL_NAME = "gpt2"
QUERY = "What is Verkada access control?"
CHECKPOINT_PATH = "./fine_tuned_verkada_gpt2"


def main():
    """
    The main driver function for the GPT-2 model.

    This function orchestrates the entire workflow, including scraping
    data from websites, loading or initializing the model and tokenizer,
    training the model, and testing it with a custom query. It handles
    the logic for using a backup dataset and loading from a checkpoint
    if available, while also providing a graceful exit on keyboard
    interruption.

    Args:
        None
    """
    try:
        # Step 1: Scrape websites for data
        if not USE_BACKUP:
            scrape_and_save()

        # Step 2: Load the model and tokenizer (will be used for fine-tuning after training)
        if exists(CHECKPOINT_PATH):
            # Load the partially-trained model
            print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
            model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_PATH)
            tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT_PATH)
        else:
            # Start training a new model
            print(
                f"No checkpoint found. Initializing from base model: {MODEL_NAME}"
            )
            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

        tokenizer.pad_token = tokenizer.eos_token

        # Step 3: Start the training process (this will fine-tune the model)
        if TRAIN_CHAT:
            print("Training chat-like behavior and developing personality.")

            # Can use personachat, dailydialog, squad, and/or multi_woz_v22
            chat_train.fine_tune_chatbot(
                model, tokenizer, ["AlekseyKorshuk/persona-chat"]
            )
            model = GPT2LMHeadModel.from_pretrained(
                "./fine_tuned_verkada_gpt2"
            )
            tokenizer = GPT2Tokenizer.from_pretrained(
                "./fine_tuned_verkada_gpt2"
            )
        else:
            print("Skipping chat training.")

        if V_TRAIN:
            print("Training on Verkada.")
            train_model(
                model,
                tokenizer,
                (
                    "verkada_data_backup.txt"
                    if USE_BACKUP
                    else "verkada_data.txt"
                ),
            )
        else:
            print("Skipping product training")

            # Step 4: After training, load the fine-tuned model
            model = GPT2LMHeadModel.from_pretrained(
                "./fine_tuned_verkada_gpt2"
            )
            tokenizer = GPT2Tokenizer.from_pretrained(
                "./fine_tuned_verkada_gpt2"
            )

        # Step 5: Test the fine-tuned model using the custom query
        response = app.test_gpt2_query(
            model, tokenizer, QUERY
        )  # Use the fine-tuned model and tokenizer
        print(f"Response: {response}")
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
