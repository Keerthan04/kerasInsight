from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from scraper.web_scraper import WebScraper
from database.pinecone_manager import PineconeManager
from embedding.text_embedder import TextEmbedder
from retrieval.information_retriever import InformationRetriever
from llm.llm_responder import LLMResponder
from evaluation.rag_evaluator import RAGEvaluator
from config import API_KEY, INDEX_NAME, MODEL_NAME, LLM_MODEL_NAME, LLM_MODEL_DIR, OPENAI_API_KEY
import asyncio

router = APIRouter()

class URLInput(BaseModel):
    urls: List[str]

class QueryInput(BaseModel):
    query: str

@router.post("/scrape")
async def scrape_endpoints(url_input: URLInput):
    pinecone_manager = PineconeManager(API_KEY, INDEX_NAME)
    pinecone_manager.initialize_index()
    embedder = TextEmbedder(MODEL_NAME)
    
    async with aiohttp.ClientSession() as session:
        tasks = [WebScraper.scrape_page(session, url, embedder, pinecone_manager.index) for url in url_input.urls]
        scraped_data = await asyncio.gather(*tasks)
    
    return {"message": "Scraping completed", "data": scraped_data}

@router.post("/query")
async def query_information(query_input: QueryInput):
    pinecone_manager = PineconeManager(API_KEY, INDEX_NAME)
    embedder = TextEmbedder(MODEL_NAME)
    retriever = InformationRetriever(pinecone_manager.index, embedder)
    
    results = await retriever.query_index(query_input.query)
    
    llm_responder = LLMResponder(LLM_MODEL_NAME, LLM_MODEL_DIR)
    
    prompt = construct_prompt(results, query_input.query)
    response = llm_responder.generate_response(prompt)
    
    # Evaluate the response
    evaluator = RAGEvaluator(OPENAI_API_KEY)
    evaluation = evaluator.evaluate(query_input.query, prompt, response)
    
    return {
        "query": query_input.query,
        "results": results,
        "llm_response": response,
        "evaluation": evaluation
    }

def construct_prompt(results, query):
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