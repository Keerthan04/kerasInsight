from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class LLMResponder:
    def __init__(self, model_name: str, model_dir: str = "./saved_model"):
        self.model_dir = model_dir
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            # Load model and tokenizer from Hugging Face Hub
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            # Save the model and tokenizer locally
            self.save_model_and_tokenizer()
        else:
            # Load model and tokenizer from the local directory
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir)

    def save_model_and_tokenizer(self):
        """Save the model and tokenizer to the specified directory."""
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    def generate_response(self, prompt: str) -> str:
        """Generate a response from the model given a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,  # Adjust as necessary
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response