# pylint: disable=redefined-outer-name

from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from threading import Lock
from time import sleep

import app.scrape  # Importing the scrape functionality
import torch
from datasets import Dataset
from tqdm import tqdm

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# Check if MPS is available on Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token  # Setting pad_token to eos_token

# Make sure the model is loaded with the correct configuration
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.to(device)  # Move model to MPS or CPU


class PauseTrainingCallback(TrainerCallback):
    def __init__(self, pause_after_steps, pause_duration):
        """
        Callback to pause training after a specified number of steps.

        Args:
            pause_after_steps (int): Number of steps after which to pause training.
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
            print(
                f"Pausing training at step {state.global_step} for {self.pause_duration}s to cool down."
            )
            sleep(self.pause_duration)
        return control


# Function to scrape the data and save to text file
def scrape_and_save():
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
    app.scrape.scrape_website(
        urls, visited_urls, lock
    )  # Calling the scrape function to save data
    print("Scraping complete, data saved to verkada_data.txt.")


# Function to split the text into chunks of max_length
def chunk_text(text, max_size=1024):
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


# Tokenize the dataset (GPT-2 requires text to be tokenized)
def tokenize_function(examples):
    inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=1024,
    )

    # Labels are the same as input_ids
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


# Function to train the model
def train_model(model, tokenizer, file_name):
    # Load the scraped data
    with open(file_name, "r", encoding="utf-8") as file:
        text_data = file.read()  # Read the entire dataset as a single string

    # Split the text into smaller documents by double newlines
    documents = text_data.split("\n\n")

    # Chunk the documents into smaller pieces
    chunked_docs, results = [], []
    print("Documents that exceed the max length will be split further.")
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        results = list(
            tqdm(
                executor.map(chunk_text, documents),
                total=len(documents),
                desc="Processing documents",
            )
        )

    for result in results:
        chunked_docs.extend(result)

    # Create a dataset from the chunked documents
    print(
        f"Total chunks created: {len(chunked_docs)}"
    )  # Debugging number of chunks created
    dataset = Dataset.from_dict({"text": chunked_docs})

    # Tokenize the dataset
    dataset = dataset.map(tokenize_function, batched=True)

    # Ensure the dataset contains the correct columns
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

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
    )

    pause_callback = PauseTrainingCallback(
        pause_after_steps=100, pause_duration=30
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[pause_callback],  # pause to allow cool down period
    )

    # Fine-tune the model
    print(f"Starting training on {device}...")
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_verkada_gpt2")
    tokenizer.save_pretrained("./fine_tuned_verkada_gpt2")
