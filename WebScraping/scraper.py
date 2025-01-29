import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
import time
from collections import defaultdict
import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Set, Dict, List, Optional
import re

class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 100, delay: float = 1.0):
        """
        Initialize the web crawler.
        
        Args:
            base_url (str): The starting URL to crawl
            max_pages (int): Maximum number of pages to crawl
            delay (float): Delay between requests in seconds
        """
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.data_dict: Dict[str, List[str]] = defaultdict(list)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.netloc == self.domain
        except Exception:
            return False

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and special characters."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def extract_page_data(self, url: str, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Extract relevant data from the page.
        
        Args:
            url (str): The URL being processed
            soup (BeautifulSoup): Parsed HTML content
            
        Returns:
            Dict[str, str]: Dictionary containing extracted data
        """
        data = {}

        # Extract title
        title = soup.title.string if soup.title else ''
        data['Title'] = self.clean_text(title)

        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        data['Meta Description'] = self.clean_text(
            meta_desc['content']) if meta_desc else ''

        # Extract main content (modify selectors based on website structure)
        main_content = soup.find('main') or soup.find(
            'article') or soup.find('div', class_='content')
        if main_content:
            data['Main Content'] = self.clean_text(main_content.get_text())

        # Extract headings
        headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])]
        data['Headings'] = '|'.join(self.clean_text(h) for h in headings)

        return data

    def process_page(self, url: str) -> Optional[Dict[str, str]]:
        """Process a single page: fetch content and extract data."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            page_data = self.extract_page_data(url, soup)

            # Find all links on the page
            links = soup.find_all('a', href=True)
            for link in links:
                full_url = urljoin(url, link['href'])
                if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                    self.data_dict['URLs'].append(full_url)

            return page_data

        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return None

    def crawl(self):
        """Main crawling method."""
        self.data_dict['URLs'].append(self.base_url)
        processed_data = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            while self.data_dict['URLs'] and len(self.visited_urls) < self.max_pages:
                current_url = self.data_dict['URLs'].pop(0)

                if current_url in self.visited_urls:
                    continue

                self.visited_urls.add(current_url)
                self.logger.info(f"Processing: {current_url}")

                page_data = executor.submit(
                    self.process_page, current_url).result()
                if page_data:
                    page_data['URL'] = current_url
                    processed_data.append(page_data)

                time.sleep(self.delay)

        return processed_data

    def save_to_csv(self, data: List[Dict[str, str]], filename: str = 'scraped_data_webcrawl.csv'):
        """Save the scraped data to a CSV file."""
        if not data:
            self.logger.warning("No data to save!")
            return

        try:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8')
            self.logger.info(f"Data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")


def main():
    # Example usage
    # Replace with your target website
    base_url = "https://pt.wikipedia.org/wiki/Brasil"
    crawler = WebCrawler(
        base_url=base_url,
        max_pages=50,  # Adjust based on your needs
        delay=1.0  # Adjust based on website's robots.txt
    )

    scraped_data = crawler.crawl()
    crawler.save_to_csv(scraped_data)


if __name__ == "__main__":
    main()
