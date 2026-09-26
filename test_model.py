from gpt4all import GPT4All
import os

MODEL_DIR = r"C:\local-chatbot\models"
MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_0.gguf"

print(f"Loading model from {MODEL_DIR}")
print(f"Model file: {MODEL_FILE}")

try:
    model = GPT4All(model_name=MODEL_FILE, model_path=MODEL_DIR)
    print("✓ Model loaded")
    
    # Test generate with only max_tokens
    response = model.generate("Hello, how are you?", max_tokens=50)
    print(f"Response: {response}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()