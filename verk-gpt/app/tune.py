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
from os import cpu_count
from threading import Lock
from time import sleep

import app.scrape as scrape  # Importing the scrape functionality

import evaluate
import torch
from app.preprocess_data import preprocess_custom_data

from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


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


class PauseTrainingCallback(TrainerCallback):
    """
    Callback to pause training after a specified number of steps.

    This callback allows the training process to pause for a defined
    duration after a certain number of steps have been completed. It is
    useful for managing resource usage and preventing overheating during
    long training sessions.

    Args:
        pause_after_steps (int): Number of steps after which to pause
            training.
        pause_duration (int): Duration of the pause in seconds.

    Methods:
        on_step_end: Pauses training if the current step is a multiple of
        the specified pause_after_steps.
    """

    def __init__(self, pause_after_steps, pause_duration):
        """
        Callback to pause training after a specified number of steps.

        Args:
            pause_after_steps (int): Number of steps after which to pause
                training.
            pause_duration (int): Duration of the pause in seconds.
        """
        self.pause_after_steps = pause_after_steps
        self.pause_duration = pause_duration

    def on_step_end(self, args, state, control, **kwargs):
        # Pause after the specified number of steps
        if (
            state.global_step % self.pause_after_steps == 0
            and state.global_step > 0
        ):
            # Uncomment for added verbosity
            # print(
            #     f"Pausing training at step {state.global_step} for "
            #     f"{self.pause_duration}s to cool down."
            # )
            sleep(self.pause_duration)
        return control


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
    # initialize_metrics
    exact_match = evaluate.load("exact_match")
    f1 = evaluate.load("f1")

    preds = p.predictions
    labels = p.label_ids

    # Decode the predicted and label token IDs to text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip leading/trailing whitespaces from the predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute Exact Match (EM) and F1 score
    em_score = exact_match.compute(
        predictions=decoded_preds, references=decoded_labels
    )
    f1_score = f1.compute(predictions=decoded_preds, references=decoded_labels)

    return {"exact_match": em_score["exact_match"], "f1": f1_score["f1"]}


def train_model(model, tokenizer, file_name):
    """
    Trains a model using text data from a specified file.

    This function loads text data, processes it into manageable chunks,
    and fine-tunes the provided model using the processed dataset. It
    sets up training parameters, initializes a trainer, and saves the
    fine-tuned model and tokenizer after training.

    Args:
        model (PreTrainedModel): The model to be trained.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with
            the model.
        file_name (str): The path to the file containing the training data.

    Returns:
        PreTrainedModel: The custom-trained transformer.

    Examples:
        train_model(my_model, my_tokenizer, "data.txt")
    """
    model, device_name = set_torch_device(model=model)

    squad_dataset, chunks = preprocess_custom_data(
        file_name, tokenizer, model, device_name
    )

    if device_name == "cuda":
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",  # Output directory for model checkpoints
            num_train_epochs=3,  # Number of training epochs
            gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
            per_device_train_batch_size=1,  # Batch size per device during training
            per_device_eval_batch_size=1,  # Batch size for evaluation
            warmup_steps=500,  # Number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # Weight decay strength
            logging_dir="./logs",  # Directory for storing logs
            save_steps=500,  # Save model every 500 steps
            fp16=True,  # Enable mixed precision (only works on supported hardware)
            save_total_limit=2,  # Keep only the last two checkpoints
            dataloader_num_workers=cpu_count(),  # Number of workers for data loading
            load_best_model_at_end=True,
            run_name="Verkada Model",
            eval_strategy="epoch",  # Evaluate after each epoch
        )

        pause_callback = PauseTrainingCallback(
            pause_after_steps=10000, pause_duration=15
        )
    else:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",  # Output directory for model checkpoints
            num_train_epochs=3,  # Number of training epochs
            gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
            per_device_train_batch_size=1,  # Batch size per device during training
            per_device_eval_batch_size=1,  # Batch size for evaluation
            warmup_steps=500,  # Number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # Weight decay strength
            logging_dir="./logs",  # Directory for storing logs
            save_steps=500,  # Save model every 500 steps
            save_total_limit=2,  # Keep only the last two checkpoints
            dataloader_num_workers=cpu_count(),  # Number of workers for data loading
            load_best_model_at_end=True,
            run_name="Verkada Model",
            eval_strategy="epoch",  # Evaluate after each epoch
        )

        pause_callback = PauseTrainingCallback(
            pause_after_steps=500, pause_duration=60
        )

    train_dataset = squad_dataset["train"]
    validation_dataset = squad_dataset["validation"]

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        callbacks=[pause_callback],  # pause to allow cool down period
    )

    # Fine-tune the model
    print(f"Starting training on {device_name}...")
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_verkada_gpt2")
    tokenizer.save_pretrained("./fine_tuned_verkada_gpt2")

    return model, chunks
