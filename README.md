# KerasInsight

## Overview

KerasInsight is a tool designed to facilitate the exploration and understanding of Keras documentation. By scraping, processing, and embedding content from Keras, users can interactively query and retrieve relevant information, including code snippets and summaries, through a user-friendly interface.

## Features

- **Web Scraping:** Extracts content from Keras documentation, organizing text and code into structured JSON files.
- **Data Processing:** Preprocesses scraped data by chunking it into meaningful sections that combine text and code for embedding.
- **Vector Database:** Stores the processed data in Pinecone for efficient similarity search.
- **LLM Integration:** Utilizes large language models to answer user queries by retrieving and processing relevant information from the vector database.
- **User Interface:** A Next.js-based interface that allows users to query the knowledge base through a chat-like window, enabling dynamic and interactive exploration.

## Project Workflow

### Scraping Keras Documentation

- Users can specify endpoints to scrape.
- The content is separated into chunks, each containing a main title, section titles, text, code, and summaries.

### Data Embedding

- The combined text and code are embedded using a Hugging Face model and stored in Pinecone.

### Querying

- Users interact with the system through a chat interface.
- Queries are processed using similarity search to retrieve relevant information from the vector database.
- If relevant information is found, the LLM generates a response; otherwise, users are informed of the absence of relevant data.

