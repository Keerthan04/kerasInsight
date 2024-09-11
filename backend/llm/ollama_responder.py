import requests
from config import OLLAMA_API_BASE

class OllamaLLMResponder:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.api_base = OLLAMA_API_BASE

    def generate_response(self, prompt: str) -> str:
        
        """
        Generate a response from the OLLAMA model based on the given prompt.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.

        Raises:
            requests.exceptions.RequestException: If there was an error making the request to the OLLAMA API.
        """

        url = self.api_base
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            
            response_json = response.json()
            actual_response = response_json.get('response', 'No response field found')
            return actual_response
        except requests.exceptions.RequestException as e:
            return f"Error: Unable to generate response. Exception: {str(e)}"

    def construct_prompt(self, results, query):
        #TODO
        #the different and good prompt has to be taught of
        """
        Construct a prompt for the LLM based on the results and query.

        The prompt is constructed as follows:

        - A general instruction to the LLM explaining its task.
        - The title, summary, and code snippets from the search results.
        - The user query.

        :param results: The search results to include in the prompt.
        :param query: The user query to include in the prompt.
        :return: The constructed prompt.
        """
        prompt = (
            "You are an AI assistant designed to provide detailed and accurate explanations based on provided information. "
            "Your task is to explain concepts clearly by synthesizing the information given below. "
            "Please ensure that the response is coherent, relevant to the user query, and incorporates the examples provided. "
            "If necessary, make logical connections between different pieces of information to create a comprehensive explanation.\n\n"
        )
        
        for result in results:
            prompt += (
                f"Title: {result['title']}\n"
                f"Summary: {result['summary']}\n\n"
                f"Code: {result['code']}\n\n"
            )
        
        prompt += f"User Query: {query}\n\n"
        return prompt