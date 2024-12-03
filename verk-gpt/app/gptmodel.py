import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 Medium
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


# Test query
def test_gpt2_query(query):
    # Encode input query
    inputs = tokenizer.encode(query, return_tensors="pt")

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Test the model with a simple query
query = "Explain quantum physics to me like I'm a 5-year old."
response = test_gpt2_query(query)
print(f"Response: {response}")
