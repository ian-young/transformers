import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import app.scrape  # Importing the scrape functionality

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


# Function to scrape the data and save to text file
def scrape_and_save():
    print("Scraping websites to gather data...")
    visited_urls = set()
    urls = [
        "https://help.verkada.com",  # Main help site
        "https://docs.verkada.com",  # Product docs
        "https://apidocs.verkada.com",  # API docs
        "https://verkada.com",  # Home page
        "https://verkada.com/pricing",  # Pricing page
    ]
    app.scrape.scrape_website(
        urls, visited_urls
    )  # Calling the scrape function to save data
    print("Scraping complete, data saved to verkada_data.txt.")


# Function to split the text into chunks of max_length
def chunk_text(text, chunk_size=512):
    # Tokenize the text into tokens
    tokens = tokenizer.encode(text)
    # Ensure that chunks are of size chunk_size or greater
    if len(tokens) < chunk_size:
        print(
            f"Warning: Text is too short, padding it to {chunk_size} tokens."
        )
        tokens += [tokenizer.pad_token_id] * (
            chunk_size - len(tokens)
        )  # pad to max length

    # Split the tokens into chunks of 512 tokens
    return [
        tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)
    ]


# Tokenize the dataset (GPT-2 requires text to be tokenized)
def tokenize_function(examples):
    chunked_texts = chunk_text(examples["text"], chunk_size=512)
    if len(chunked_texts) == 0:
        print("Warning: Encountered empty token chunks.")
    input_ids = [chunk for chunk in chunked_texts]
    return {"input_ids": input_ids, "labels": input_ids}


# Function to train the model
def train_model(model, tokenizer):
    # Load the scraped data
    with open("verkada_data.txt", "r", encoding="utf-8") as file:
        text_data = file.read()

    # Check if MPS is available on Apple Silicon
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} for training.")

    model.to(device)  # Move model to MPS or CPU

    # Prepare dataset from the scraped data, splitting the text into smaller chunks
    dataset = Dataset.from_dict({"text": [text_data]})
    dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        save_steps=500,  # Save model every 500 steps
        save_total_limit=2,  # Keep only the last two checkpoints
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model after fine-tuning
    model.save_pretrained("./fine_tuned_verkada_gpt2")
    tokenizer.save_pretrained("./fine_tuned_verkada_gpt2")
