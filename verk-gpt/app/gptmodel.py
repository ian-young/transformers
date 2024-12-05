"""
Author: Ian Young

This module provides functionality to interact with a GPT-2 model for 
generating text responses based on input queries.

It includes a function to generate responses from the GPT-2 model and
handles the loading of a pre-trained model and tokenizer. The module
can either load a fine-tuned model from a checkpoint or initialize a
new model from a specified base model.

Functions:
    test_gpt2_query: Generates a response from a GPT-2 model based on a
        given input query.

Usage:
    The module can be run as a standalone script to test the model with a
    predefined query. It will load the appropriate model and tokenizer,
    generate a response, and print it to the console.
"""

# pylint: disable=redefined-outer-name

from os.path import exists

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Test query function
def test_gpt2_query(model, tokenizer, query):
    """
    Generates a response from a GPT-2 model based on a given input query.

    This function encodes the input query, prepares the necessary attention
    mask, and uses the model to generate a response. It ensures that the
    output is coherent and avoids repetitive phrases by applying various
    sampling techniques.

    Args:
        model (PreTrainedModel): The GPT-2 model used for generating
            responses.
        tokenizer (PreTrainedTokenizer): The tokenizer associated
            with the model.
        query (str): The input query for which a response is to be
            generated.

    Returns:
        str: The generated response from the model.

    Examples:
        response = test_gpt2_query(
            model,
            tokenizer,
            "What is the capital of France?"
        )
    """
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
            no_repeat_ngram_size=3,  # Avoid reptitive phrases
            temperature=0.6,  # Lower for deterministic. Higher for creativity.
            top_k=40,  # Consider top 40 tokens each step
            top_p=0.9,  # Use nucleus sampling for natural output
            do_sample=True,  # Enable sampling for creativity
            repetition_penalty=1.2,  # Penalize repeated tokens
            length_penalty=1.1,  # Encourage shorter responses
            num_beams=4,  # Use 4 beams to consider outputs
        )

    # Decode the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Load the default GPT-2 model and tokenizer
    MODEL_NAME = "gpt2-medium"
    if exists("./fine_tuned_verkada_gpt2"):
        # Load the partially-trained model
        print("Loading model from checkpoint: ./fine_tuned_verkada_gpt2")
        model = GPT2LMHeadModel.from_pretrained("./fine_tuned_verkada_gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_verkada_gpt2")
    else:
        # Start training a new model
        print(
            f"No checkpoint found. Initializing from base model: {MODEL_NAME}"
        )
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # Test the model with a simple query
    QUERY = "Explain the difference between the CD52 and CD42."
    response = test_gpt2_query(model, tokenizer, QUERY)
    print(f"Response: {response}")
