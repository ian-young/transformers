from os.path import exists
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from app.tune import train_model_with_dataset

# List of datasets related to personality and chat-specific training
PERSONALITY_CHAT_DATASETS = [
    "AlekseyKorshuk/persona-chat",  # Personality chat dataset
    "roskoN/dailydialog",  # Daily dialogues dataset
    "pfb30/multi_woz_v22",  # Multi-domain task-oriented dialogues
    "rajpurkar/squad",  # Q&A format
]


def fine_tune_chatbot(model, tokenizer, dataset_names):
    """
    Fine-tunes the chatbot using specified datasets.
    Args:
        dataset_names (list of str): List of dataset names to use for training.
    """
    for dataset_name in dataset_names:
        print(f"Fine-tuning with dataset: {dataset_name}")
        train_model_with_dataset(model, tokenizer, dataset_name)


if __name__ == "__main__":
    MODEL_NAME = "gpt2-medium"
    if exists("./fine_tuned_verkada_gpt2"):
        # Load the partially-trained model
        print("Loading model from checkpoint: ./fine_tuned_verkada_gpt2")
        gpt_model = GPT2LMHeadModel.from_pretrained(
            "./fine_tuned_verkada_gpt2"
        )
        gpt_tokenizer = GPT2Tokenizer.from_pretrained(
            "./fine_tuned_verkada_gpt2"
        )
    else:
        # Start training a new model
        print(
            f"No checkpoint found. Initializing from base model: {MODEL_NAME}"
        )
        gpt_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        gpt_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    # You can loop over datasets or manually choose one for training
    for dataset in PERSONALITY_CHAT_DATASETS:
        fine_tune_chatbot(gpt_model, gpt_tokenizer, dataset)
