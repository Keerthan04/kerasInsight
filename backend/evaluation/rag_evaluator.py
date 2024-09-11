from trulens_eval import Feedback, TruLlama, OpenAI
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as OpenAIProvider

class RAGEvaluator:
    def __init__(self, openai_api_key):
        self.openai = OpenAI(api_key=openai_api_key)
        self.groundedness = Groundedness(groundedness_provider=self.openai)

        # Initialize feedback functions
        self.f_contextual_relevance = Feedback(self.openai.relevance_with_cot_reasons, name="Contextual Relevance").on_input_output()
        self.f_answer_relevance = Feedback(self.openai.relevance_with_cot_reasons, name="Answer Relevance").on_input_output()
        self.f_groundedness = Feedback(self.groundedness.grounded_statements_aggregated_with_reasons, name="Groundedness").on_output()

    def evaluate(self, query, context, response):
        # Create a TruLlama recorder (normally used with a Llama index, but we'll use it for structuring our evaluation)
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
            "reasons": {
                "contextual_relevance": results[self.f_contextual_relevance.name].reason,
                "answer_relevance": results[self.f_answer_relevance.name].reason,
                "groundedness": results[self.f_groundedness.name].reason
            }
        }