from transformers import GPT2LMHeadModel, GPT2Tokenizer
from app.tune import train_model  # Importing the function to start training
import app

# Step 1: Load the model and tokenizer (will be used for fine-tuning after training)
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 2: Start the training process (this will fine-tune the model)
train_model(model, tokenizer)  # Function that runs the training

# Step 3: After training, load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_verkada_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_verkada_gpt2")

# Step 4: Test the fine-tuned model
query = "What is Verkada access control?"
response = app.test_gpt2_query(query)  # Testing the query function
print(f"Response: {response}")
