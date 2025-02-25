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

from app.tune import (  # Importing the function to start training
    scrape_and_save,
    train_model,
)

USE_BACKUP = False
GENERATE_SQUAD_DATA = True
V_TRAIN = True


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

        if V_TRAIN:
            print("Training on Verkada.")
            file_name = (
                "verkada_data_backup.txt" if USE_BACKUP else "verkada_data.txt"
            )

            train_model(
                file_name,
                GENERATE_SQUAD_DATA,
            )
        else:
            print("Skipping product training")

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
