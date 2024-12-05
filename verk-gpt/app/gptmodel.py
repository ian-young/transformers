# pylint: disable=redefined-outer-name

from os.path import exists

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Test query function
def test_gpt2_query(model, tokenizer, query):
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token
    # Encode the input query
    inputs = tokenizer.encode(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Generate attention mask: 1 for actual tokens, 0 for padding tokens
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,  # Pass the attention mask here
            num_return_sequences=1,
            max_length=512,  # Allow for longer responses
            no_repeat_ngram_size=2,  # Avoid reptitive phrases
            temperature=0.7,  # Adjust randomness
            top_k=50,  # Consider top 50 tokens each step
            top_p=0.9,  # Use nucleus sampling for natural output
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # Load the default GPT-2 model and tokenizer
    model_name = "gpt2-medium"
    if exists("./fine_tuned_verkada_gpt2"):
        # Load the partially-trained model
        print(f"Loading model from checkpoint: {"./fine_tuned_verkada_gpt2"}")
        model = GPT2LMHeadModel.from_pretrained("./fine_tuned_verkada_gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_verkada_gpt2")
    else:
        # Start training a new model
        print(
            f"No checkpoint found. Initializing from base model: {model_name}"
        )
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Test the model with a simple query
    query = "Explain quantum physics to me like I'm a 5-year old."
    response = test_gpt2_query(model, tokenizer, query)
    print(f"Response: {response}")
