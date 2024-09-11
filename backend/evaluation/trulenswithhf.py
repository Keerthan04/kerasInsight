from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback import Groundedness
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Any, Callable, Dict, Optional, Union

class HuggingFaceProvider:
    def __init__(self, model_name: str = "cross-encoder/qnli-distilroberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def relevance(self, query: str, response: str) -> float:
        inputs = self.tokenizer(query, response, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        return scores[:, 1].item()  # Return the score for the "entailment" class

    def grounded_statements(
        self,
        statements: Union[str, list],
        context: str,
    ) -> Dict[str, float]:
        if isinstance(statements, str):
            statements = [statements]
        
        results = {}
        for statement in statements:
            score = self.relevance(context, statement)
            results[statement] = score
        
        return results

class RAGEvaluator:
    def __init__(self, model_name: str = "cross-encoder/qnli-distilroberta-base"):
        self.hf_provider = HuggingFaceProvider(model_name)
        self.groundedness = Groundedness(groundedness_provider=self.hf_provider)

        # Initialize feedback functions
        self.f_contextual_relevance = Feedback(self.hf_provider.relevance, name="Contextual Relevance").on_input_output()
        self.f_answer_relevance = Feedback(self.hf_provider.relevance, name="Answer Relevance").on_input_output()
        self.f_groundedness = Feedback(self.groundedness.grounded_statements, name="Groundedness").on_input_output()

    def evaluate(self, query: str, context: str, response: str):
        # Create a TruLlama recorder
        tru_recorder = TruLlama(
            app_id="RAG_Evaluation",
            feedbacks=[
                self.f_contextual_relevance,
                self.f_answer_relevance,
                self.f_groundedness
            ]
        )

        # Record the evaluation
        with tru_recorder as recording:
            recording.input = {"query": query, "context": context}
            recording.output = response

        # Get the evaluation results
        results = recording.evaluate()

        return {
            "contextual_relevance": results[self.f_contextual_relevance.name].score,
            "answer_relevance": results[self.f_answer_relevance.name].score,
            "groundedness": results[self.f_groundedness.name].score,
            "overall_score": (results[self.f_contextual_relevance.name].score +
                              results[self.f_answer_relevance.name].score +
                              results[self.f_groundedness.name].score) / 3
        }

# Example usage
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
    response = "The capital of France is Paris."
    
    results = evaluator.evaluate(query, context, response)
    print(results)
    
'''
Update requirements.txt to include:
Copytrulens-eval
transformers
torch

Replace the content of evaluation/rag_evaluator.py with the code provided in the artifact.
Update app/routes.py to use this new evaluator:
pythonCopyfrom evaluation.rag_evaluator import RAGEvaluator

# In the query_information function:
evaluator = RAGEvaluator()
evaluation = evaluator.evaluate(query_input.query, context, response)
'''