import google.generativeai as genai
from litellm import completion
from trulens.feedback.llm_provider import LLMProvider
from trulens.providers.litellm import LiteLLM
from trulens.core.utils.serial import Lens
from trulens.core.feedback import Feedback
import numpy as np

class GeminiRAGEvaluator:
    def __init__(self, gemini_api_key, context: str):
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        print("type of context is \n",type(context))
        self.context = context#convert the context to lens type for evaluation
        # Set up LiteLLM provider for Gemini
        self.gemini_provider = LiteLLM(provider="gemini")
        # Initialize feedback functions
        
        #shd try this
        # self.context_relevance = (
        #     Feedback(self.gemini_provider.context_relevance_with_cot_reasons)
        #     .on_input()
        #     .on(self.context)
        #     .aggregate(np.mean)
        # )
        
        self.f_qa_relevance = (
            Feedback(self.gemini_provider.relevance_with_cot_reasons)
            .on_input()
            .on_output()
        )

    def query_gemini(self, prompt):
        # Query the Gemini model and get the response
        response = self.model.generate_content(prompt)
        return response.text

    def evaluate(self, query, context, response):
        # Custom evaluation logic without TruLlama
        # context_relevance_score = self.context_relevance(query, context, response)
        answer_relevance_score = self.f_qa_relevance(query, response)

        # Return the evaluation results
        return {
            "answer_relevance": answer_relevance_score
        }
