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
            attention_mask=attention_mask,  # Pass the attention mask here
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



if __name__ == "__main__":
    # Load the default GPT-2 model and tokenizer
    model_name = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Test the model with a simple query
    query = "Explain quantum physics to me like I'm a 5-year old."
    response = test_gpt2_query(model, tokenizer, query)
    print(f"Response: {response}")
