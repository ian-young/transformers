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
from app.tune import (  # Importing the function to start training
    scrape_and_save,
    train_model,
)

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

USE_BACKUP = True
GENERATE_SQUAD_DATA = True
V_TRAIN = True
QUERY = "How much power does a CD42 draw?"
CHECKPOINT_PATH = "./fine_tuned_verkada"


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
        checkpoint_path = "./fine_tuned_verkada"

        if exists(checkpoint_path):
            print(
                f"Loading model and tokenizer from checkpoint: {checkpoint_path}"
            )
            qa_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            # Step 2: Load the model and tokenizer (will be used for fine-tuning after training)
            qa_model_name = "potsawee/t5-large-generation-squad-QuestionAnswer"
            print(
                f"No checkpoint found. Initializing from base model: {qa_model_name}"
            )
            qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)
            tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

        file_name = (
            "verkada_data_backup.txt" if USE_BACKUP else "verkada_data.txt"
        )

        qa_model, device = app.set_torch_device(qa_model)
        tokenizer.pad_token = tokenizer.eos_token

        if V_TRAIN:
            print("Training on Verkada.")
            model, chunks = train_model(
                qa_model,
                device,
                tokenizer,
                file_name,
                GENERATE_SQUAD_DATA,
            )
        else:
            print("Skipping product training")
            model = qa_model
            chunks = []

        # Step 4: Querying example (for inference using retriever and QA model)
        relevant_chunks = app.retrieve(
            query=QUERY,
            chunks=chunks,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        # Choose the most relevant chunk (for simplicity)
        context = relevant_chunks["data"]
        input_text = f"question: {QUERY} context: {context}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        # Generate the answer
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {answer}\nSee {relevant_chunks['url']}")
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
