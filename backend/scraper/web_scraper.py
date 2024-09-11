import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict
from pinecone import Pinecone
from embedding.text_embedder import TextEmbedder
from config import NAMESPACE

class WebScraper:
    def __init__(self, session: aiohttp.ClientSession, embedder: TextEmbedder, index: Pinecone):
        self.session = session
        self.embedder = embedder
        self.index = index

    @staticmethod
    async def extract_sections(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extracts sections from a parsed HTML page.

        Sections are defined as a block of text or code that is separated by a heading
        (h1, h2, or h3). The function returns a list of dictionaries, where each dictionary
        contains the title of the section and the content of the section as a string. The
        content string contains the text and code blocks in the section, separated by
        newlines.

        Args:
            soup: A parsed HTML page.

        Returns:
            A list of dictionaries, where each dictionary contains the title and content of
            a section.
        """
        sections = []
        current_section = {"title": "Introduction", "content": "", "code": ""}

        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'pre']):
            if element.name in ['h1', 'h2', 'h3']:
                if current_section["content"] or current_section["code"]:
                    sections.append(current_section)
                current_section = {"title": element.get_text(strip=True), "content": "", "code": ""}
            elif element.name == 'p':
                current_section["content"] += element.get_text(strip=True) + "\n"
            elif element.name == 'pre':
                code = element.find('code')
                if code:
                    current_section["code"] += code.get_text(strip=True) + "\n\n"

        if current_section["content"] or current_section["code"]:
            sections.append(current_section)

        return sections

    @staticmethod
    def summarize_text(text: str, max_length: int = 200) -> str:
        """Summarizes a given text by returning the first `max_length` words.

        If the text is longer than `max_length`, an ellipsis is appended to the end of
        the returned string to indicate that the text has been truncated.

        Args:
            text (str): The text to be summarized.
            max_length (int, optional): The maximum number of words to return. Defaults to 200.

        Returns:
            str: The summarized text.
        """
        words = text.split()
        return ' '.join(words[:max_length]) + ('...' if len(words) > max_length else '')

    async def scrape_page(self, url: str) -> Dict[str, any]:
        """
        Scrapes a single page, extracts sections from the page, embeds the sections using a
        TextEmbedder, and upserts the embedded vectors into a Pinecone index.

        Args:
            url (str): The URL of the page to scrape.

        Returns:
            A dictionary containing the scraped data. The dictionary has a single key,
            'url', which contains the URL of the page that was scraped. The value of this
            key is another dictionary containing the scraped data, which includes the
            title, content, and code of each section, as well as a summary of the content
            of each section. If there was an error scraping the page, the dictionary will
            contain an 'error' key instead, with a string describing the error.
        """
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    sections = await self.extract_sections(soup)

                    pinecone_data = []
                    for i, section in enumerate(sections):
                        section["summary"] = self.summarize_text(section["content"])
                        combined_text = f"{section['summary']}\n\nRelated Code:\n{section['code']}"
                        section["embedding"] = self.embedder.embed_text(combined_text)

                        id = f"{url}-{i}"
                        metadata = {
                            'title': section['title'],
                            'summary': section['summary'],
                            'content': section['content'],
                            'code': section['code']
                        }
                        pinecone_data.append((id, section["embedding"], metadata))

                    self.index.upsert(vectors=pinecone_data, namespace=NAMESPACE)

                    return {'url': url, 'sections': sections}
                else:
                    print(f"Failed to fetch {url}: HTTP {response.status}")
                    return {'url': url, 'error': f"HTTP {response.status}"}
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {'url': url, 'error': str(e)}

    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, any]]:
        """Scrape a list of URLs and return a list of dictionaries containing the scraped data.

        Args:
            urls (List[str]): A list of URLs to scrape.

        Returns:
            List[Dict[str, any]]: A list of dictionaries containing the scraped data.
        """
        #TODO
        #basically we dont need to return the dict so just it shd have that and scraped shd be stored in the embedded vector that is important so here returned as the post request is returning it but we dont need to mainly
        tasks = [self.scrape_page(url) for url in urls]
        return await asyncio.gather(*tasks)
