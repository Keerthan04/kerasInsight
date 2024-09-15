import google.generativeai as genai
from config import GEMINI_API_KEY
class GeminiResponder:
    def __init__(self,GEMINI_API_KEY:str = GEMINI_API_KEY) -> None:
        """
        Initialize the GeminiResponder with a given API key.

        Args:
            GEMINI_API_KEY (str, optional): The API key to use for the Gemini API. Defaults to GEMINI_API_KEY.
        """

        self.api_key = GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        # genai.llm.list_models()

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model given a prompt.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        # for model_info in genai.list_tuned_models():
        #     print(model_info.name)
        # model_info = genai.get_model("tunedModels/finetuninggemmafordl1-xxcubsl6ftaf")
        # print(model_info)
        model = genai.GenerativeModel('tunedModels/finetuninggemmafordl1-xxcubsl6ftaf')
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error querying fine-tuned model: {e}")
            return None

    def construct_prompt(self, results, query):
        """
        Construct a well-structured prompt for the LLM based on search results and the user query.

        The prompt is designed to guide the LLM in providing clear, relevant, and synthesized responses
        based on the provided search results and user query.

        :param results: A list of dictionaries containing search results (title, summary, code).
        :param query: The user's query for which the response is being generated.
        :return: A tuple of strings representing the constructed prompt and the context for the LLM.
        """

        # System message that guides the LLM on how to respond.
        system_message = (
            "You are an AI assistant specialized in explaining Keras concepts. When answering questions, follow these guidelines:\n"
            "1. Provide clear, concise explanations of key concepts.\n"
            "2. Include relevant code examples to illustrate your points.\n"
            "3. Synthesize information without repeating any part of the given context verbatim.\n"
            "4. Focus on the most important aspects related to the user's query.\n"
            "5. If appropriate, mention any limitations or common use cases.\n"
            "6. Start the response immediately with the relevant information, without restating the query or the system message.\n"
        )

        # Instruction for formatting the response.
        instruction = (
            "Please provide only the assistant's response.\n"
            "The response should start with 'Assistant:' and end with '[End of Assistant Response]'.\n"
            "Include relevant code in your response by referring to the context provided.\n"
        )

        # Initialize prompt and context parts
        prompt_parts = [system_message, "\nContext:\n"]
        context_parts = ["Context:\n"]

        # Iterate through the results to build the prompt and context with proper formatting
        for result in results:
            try:
                # Extracting result details with defaults
                title = result.get("title", 'No Title')
                summary = result.get("summary", 'No Summary')
                code = result.get("code", 'No Code Provided')

                # Ensure the values are strings to prevent formatting issues
                if not isinstance(title, str):
                    title = 'No Title'
                if not isinstance(summary, str):
                    summary = 'No Summary'
                if not isinstance(code, str):
                    code = 'No Code Provided'

                # Format the result string with proper line breaks and indentation
                result_str = (
                    f"\nTitle: {title}\n"
                    f"Summary:\n    {summary}\n"
                    f"Code:\n    {code}\n"
                )

                # Append formatted string to both prompt and context parts
                prompt_parts.append(result_str)
                context_parts.append(result_str)

            except Exception as e:
                print(f"Error processing result: {e}")

        # Add the user query and instruction at the end of the prompt
        prompt_parts.append(f"\nUser Query: {query}\n\nInstruction: {instruction}\nAssistant:")
        prompt = ''.join(prompt_parts).strip()

        # Join context parts
        context = ''.join(context_parts).strip()
        print("Debugging Context:", repr(context))
        return prompt, context



