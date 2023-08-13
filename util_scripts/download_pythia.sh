from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Define the model name or path
model_name_or_path = "EleutherAI/pythia-1b"

# Define the target directory where you want to save the model
target_directory = "./pythia-1b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# Save the model and tokenizer to the specified directory
model.save_pretrained(target_directory)
tokenizer.save_pretrained(target_directory)

print("Model and tokenizer saved to:", target_directory)
