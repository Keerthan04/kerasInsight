import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import torch
from typing import Dict, Any
from trulens_eval import TruCustomApp, Feedback, Tru

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class LocalEvaluator:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def sentiment_analysis(self, text: str) -> float:
        return self.sentiment_analyzer.polarity_scores(text)['compound']

    def coherence(self, text: str) -> float:
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0  # Assume perfect coherence for single sentences
        
        embeddings = self.sentence_transformer.encode(sentences)
        cosine_similarities = torch.nn.functional.cosine_similarity(embeddings[:-1], embeddings[1:])
        return cosine_similarities.mean().item()

    def relevance(self, query: str, response: str) -> float:
        query_embedding = self.sentence_transformer.encode([query])
        response_embedding = self.sentence_transformer.encode([response])
        similarity = torch.nn.functional.cosine_similarity(query_embedding, response_embedding)
        return similarity.item()

    def conciseness(self, text: str) -> float:
        return 1.0 / (1.0 + len(text) / 100)  # Normalize to 0-1 range

class CustomEvaluator:
    def __init__(self):
        self.local_evaluator = LocalEvaluator()
        self.tru = Tru()
        self.app = TruCustomApp(name="KerasInsight")

    def evaluate(self, query: str, response: str, context: str) -> Dict[str, Any]:
        # Define feedback functions
        sentiment = Feedback(self.local_evaluator.sentiment_analysis, name="Sentiment")
        coherence = Feedback(self.local_evaluator.coherence, name="Coherence")
        relevance = Feedback(self.local_evaluator.relevance, name="Relevance")
        conciseness = Feedback(self.local_evaluator.conciseness, name="Conciseness")

        # Record the evaluation
        with self.app as recording:
            recording.record_text(text=query, key="query")
            recording.record_text(text=response, key="response")
            recording.record_text(text=context, key="context")

        # Run feedback functions
        results = self.tru.run_feedback(
            app=self.app,
            feedbacks=[
                sentiment.on_output(),
                coherence.on_output(),
                relevance.on_input_output(),
                conciseness.on_output()
            ]
        )

        # Process results
        evaluation_results = {
            "sentiment": results[0].result,
            "coherence": results[1].result,
            "relevance": results[2].result,
            "conciseness": results[3].result
        }

        return evaluation_results