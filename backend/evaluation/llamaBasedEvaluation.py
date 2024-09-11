from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LlamaRAGEvaluator:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", model_dir="./saved_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_contextual_relevance(self, query, context):
        prompt = f"Rate the relevance of the following context to the query on a scale of 0 to 10, where 0 is completely irrelevant and 10 is highly relevant.\n\nQuery: {query}\n\nContext: {context}\n\nRelevance score:"
        response = self.generate_response(prompt)
        try:
            score = float(response.strip()) / 10
            return max(0, min(score, 1))  # Ensure score is between 0 and 1
        except ValueError:
            return 0.5  # Default to middle score if parsing fails

    def evaluate_answer_relevance(self, query, response):
        prompt = f"Rate how well the following response answers the query on a scale of 0 to 10, where 0 is completely irrelevant and 10 is a perfect answer.\n\nQuery: {query}\n\nResponse: {response}\n\nRelevance score:"
        response = self.generate_response(prompt)
        try:
            score = float(response.strip()) / 10
            return max(0, min(score, 1))  # Ensure score is between 0 and 1
        except ValueError:
            return 0.5  # Default to middle score if parsing fails

    def evaluate_groundedness(self, context, response):
        prompt = f"Rate how well the following response is grounded in the given context on a scale of 0 to 10, where 0 is not grounded at all and 10 is perfectly grounded.\n\nContext: {context}\n\nResponse: {response}\n\nGroundedness score:"
        response = self.generate_response(prompt)
        try:
            score = float(response.strip()) / 10
            return max(0, min(score, 1))  # Ensure score is between 0 and 1
        except ValueError:
            return 0.5  # Default to middle score if parsing fails

    def evaluate(self, query, context, response):
        contextual_relevance = self.evaluate_contextual_relevance(query, context)
        answer_relevance = self.evaluate_answer_relevance(query, response)
        groundedness = self.evaluate_groundedness(context, response)

        return {
            "contextual_relevance": contextual_relevance,
            "answer_relevance": answer_relevance,
            "groundedness": groundedness,
            "overall_score": (contextual_relevance + answer_relevance + groundedness) / 3
        }
        
# Example usage
if __name__ == "__main__":
    evaluator = LlamaRAGEvaluator()
    query = "What is the capital of France?"
    context = "The capital of France is Paris."
    response = "The capital of France is Paris."
    evaluation = evaluator.evaluate(query, context, response)
    print(evaluation)

'''
Replace the content of evaluation/rag_evaluator.py with the code provided in the artifact above.
Update app/routes.py:
Replace the import and initialization of the evaluator with:
pythonCopyfrom evaluation.rag_evaluator import LlamaRAGEvaluator

# In the query_information function:
evaluator = LlamaRAGEvaluator(LLM_MODEL_NAME, LLM_MODEL_DIR)

Ensure that config.py still contains the necessary Llama model information:
pythonCopyLLM_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LLM_MODEL_DIR = "./saved_model"
'''