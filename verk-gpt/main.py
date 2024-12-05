from os.path import exists

import app
from app.tune import (  # Importing the function to start training
    scrape_and_save,
    train_model,
)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

USE_BACKUP = True
MODEL_NAME = "gpt2"
QUERY = "What is Verkada access control?"
CHECKPOINT_PATH = "./fine_tuned_verkada_gpt2"


def main():
    """Driver function"""
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

        # Step 3: Start the training process (this will fine-tune the model)
        train_model(
            model,
            tokenizer,
            "verkada_data_backup.txt" if USE_BACKUP else "verkada_data.txt",
        )

        # Step 4: After training, load the fine-tuned model
        model = GPT2LMHeadModel.from_pretrained("./fine_tuned_verkada_gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_verkada_gpt2")

        # Step 5: Test the fine-tuned model using the custom query
        response = app.test_gpt2_query(
            model, tokenizer, QUERY
        )  # Use the fine-tuned model and tokenizer
        print(f"Response: {response}")
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
