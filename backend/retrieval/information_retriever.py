from typing import List, Dict
from config import NAMESPACE
from pinecone import Pinecone
from embedding.text_embedder import TextEmbedder
class InformationRetriever:
    def __init__(self, index: Pinecone, embedder: TextEmbedder):
        self.index = index
        self.embedder = embedder

    async def query_index(self, query_text: str) -> List[Dict[str, any]]:
        """
        Queries the Pinecone index with the given query text, and returns a
        list of up to 5 matching documents. Each document is represented as a
        dictionary with the following keys:

        - id: the document ID
        - score: the relevance score of the document
        - title: the title of the document (empty string if none)
        - summary: a brief summary of the document (empty string if none)
        - content: the full content of the document (empty string if none)
        - code: any code snippets extracted from the document (empty string if none)

        If an error occurs while querying the index, an empty list is returned.
        """
        query_embedding = self.embedder.embed_text(query_text)

        try:
            results = self.index.query(
                vector=query_embedding,
                namespace=NAMESPACE,
                top_k=5,
                include_metadata=True
            )
            print('Results from the query:\n', results)
            formatted_results = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                formatted_results.append({
                    'id': match.get('id'),
                    'score': match.get('score'),
                    'title': metadata.get('title', "No title"),
                    'summary': metadata.get('summary', "No summary"),
                    'content': metadata.get('content', "No content"),
                    'code': metadata.get('code', "No code")
                })

            return formatted_results

        except Exception as e:
            print(f"Error querying Pinecone: {str(e)}")
            return []