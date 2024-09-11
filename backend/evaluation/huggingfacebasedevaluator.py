from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

class HuggingFaceRAGEvaluator:
    def __init__(self):
        # Load models
        self.relevance_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        self.nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def evaluate_contextual_relevance(self, query, context):
        query_embedding = self.relevance_model.encode(query, convert_to_tensor=True)
        context_embedding = self.relevance_model.encode(context, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(query_embedding, context_embedding)
        return similarity.item()

    def evaluate_answer_relevance(self, query, response):
        result = self.qa_model(question=query, context=response)
        return result['score']

    def evaluate_groundedness(self, context, response):
        hypothesis = "This response is grounded in the given context."
        result = self.nli_model(response, [hypothesis, "This response is not grounded in the given context."], multi_label=False)
        return result['scores'][0] if result['labels'][0] == hypothesis else 1 - result['scores'][0]

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
    evaluator = HuggingFaceRAGEvaluator()
    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
    response = "The capital of France is Paris."
    
    results = evaluator.evaluate(query, context, response)
    print(results)
    

'''
Update requirements.txt:
Add the following lines:
Copytransformers
sentence-transformers
torch

Replace evaluation/rag_evaluator.py with the code provided in the artifact above.
Update app/routes.py:
Replace the import and initialization of RAGEvaluator with:
pythonCopyfrom evaluation.rag_evaluator import HuggingFaceRAGEvaluator

# In the query_information function:
evaluator = HuggingFaceRAGEvaluator()

Remove the OPENAI_API_KEY from config.py as it's no longer needed.
'''