from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from config import GEMMA_MODEL_SAVED_DIR
# class LLMResponder:
#     def __init__(self, model_name: str, model_dir: str = "./saved_model"):
#         self.model_dir = model_dir
#         if not os.path.exists(model_dir) or not os.listdir(model_dir):
#             # Load model and tokenizer from Hugging Face Hub
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForCausalLM.from_pretrained(model_name)
#             # Save the model and tokenizer locally
#             self.save_model_and_tokenizer()
#         else:
#             # Load model and tokenizer from the local directory
#             self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
#             self.model = AutoModelForCausalLM.from_pretrained(model_dir)

#     def save_model_and_tokenizer(self):
#         """Save the model and tokenizer to the specified directory."""
#         self.model.save_pretrained(self.model_dir)
#         self.tokenizer.save_pretrained(self.model_dir)

#     def generate_response(self, prompt: str) -> str:
#         """Generate a response from the model given a prompt."""
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         outputs = self.model.generate(
#             inputs["input_ids"],
#             max_length=512,  # Adjust as necessary
#             num_return_sequences=1,
#             pad_token_id=self.tokenizer.eos_token_id,
#             temperature=0.7,
#             top_p=0.9,
#             top_k=50
#         )
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response

# another class where we can get from directory the llm and then ask question based on that
class LLMResponder:
    def __init__(self,model_dir:str = GEMMA_MODEL_SAVED_DIR) -> None:
        """
        Initialize the LLMResponder with a given model directory.

        Args:
            model_dir (str, optional): The directory containing the model and tokenizer.
                Defaults to GEMMA_MODEL_SAVED_DIR.
        """

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model given a prompt.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=3000)
        response = self.tokenizer.decode(outputs[0])
        
        start_marker = "Assistant:"
        end_marker = "[End of Assistant Response]"

        start_idx = response.rfind(start_marker) + len(start_marker)
        end_idx = response.rfind(end_marker)

        if start_idx != -1 and end_idx != -1:
            assistant_response = response[start_idx:end_idx].strip()
        else:
            assistant_response = response.strip()
        return assistant_response
    def construct_prompt(self, results, query):
        """
        Construct a well-structured prompt for the LLM based on search results and the user query.

        The prompt is designed to guide the LLM in providing clear, relevant, and synthesized responses
        based on the provided search results and user query.

        :param results: A list of dictionaries containing search results (title, summary, code).
        :param query: The user's query for which the response is being generated.
        :return: A string representing the constructed prompt for the LLM.
        """

        # System message that guides the LLM on how to respond.
        system_message = """
            You are an AI assistant specialized in explaining Keras concepts. When answering questions, follow these guidelines:
            1. Provide clear, concise explanations of key concepts.
            2. Include relevant code examples to illustrate your points.
            3. Synthesize information without repeating any part of the given context verbatim.
            4. Focus on the most important aspects related to the user's query.
            5. If appropriate, mention any limitations or common use cases.
            6. Start the response immediately with the relevant information, without restating the query or the system message.
        """

        # Instruction for formatting the response.
        instruction = (
            "Please provide only the assistant's response. "
            "The response should start with 'Assistant:' and end with '[End of Assistant Response]'."
            "Include Relevent Code in Your Response by referring to the context Provided"
        )

        # Construct the core of the prompt.
        prompt = f"{system_message}\n\n"
        prompt += "Context Starts from here Each context has title summary and code:\n"
        context = "context:\n"
        # Add each result's title, summary, and code snippet to the prompt.
        for result in results:
            title = result.get('title', 'No Title')
            summary = result.get('summary', 'No Summary')
            code = result.get('code', 'No Code Provided')
            
            prompt += f"Title: {title}\n"
            prompt += f"Summary: {summary}\n"
            prompt += f"Code:\n{code}\n\n"
            context += f"Title: {title}\n"
            context += f"Summary: {summary}\n"
            context += f"Code:\n{code}\n\n"
        # Add the user query and instruction at the end.
        prompt += f"User Query: {query}\n\n"
        prompt += f"Instruction: {instruction}\n"
        prompt += "Assistant:"

        return prompt,context
