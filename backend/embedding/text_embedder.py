from sentence_transformers import SentenceTransformer
from typing import List

class TextEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        # self.model.to(device)  # Move model to GPU(in collab i tried)

    def embed_text(self, text: str) -> List[float]:
        #TODO
        #use of different model for the embedding part
        """
        Embeds a given text into a vector of floats using the stored model.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: A list of floats representing the embedded text.
        """

        return self.model.encode(text, convert_to_tensor=True).tolist()