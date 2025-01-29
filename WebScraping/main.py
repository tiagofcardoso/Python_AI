import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse
from collections import defaultdict
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Set, Dict, List
import re
import os 

dirname = os.path.dirname(__file__)

class WebScraper:
    def __init__(self, base_url: str, max_pages: int = 100):
        self.base_url = base_url
        self.domain = urllib.parse.urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.data_dict: Dict[str, List[str]] = defaultdict(list)
        self.max_pages = max_pages
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='scraping.log'
        )

    def is_valid_url(self, url: str) -> bool:
        """Validate if URL belongs to same domain and is accessible"""
        try:
            parsed = urllib.parse.urlparse(url)
            return (
                parsed.netloc == self.domain and
                any(url.endswith(ext) for ext in ['.html', '.htm', '/', ''])
            )
        except Exception:
            return False

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        return ' '.join(text.split())

    def extract_page_data(self, url: str, soup: BeautifulSoup) -> None:
        """Extract relevant data from page"""
        # Get page title
        title = soup.title.string if soup.title else url
        
        # Extract text content
        text_content = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            cleaned_text = self.clean_text(tag.get_text())
            if cleaned_text:
                text_content.append(cleaned_text)
        
        # Store data
        self.data_dict[url] = {
            'title': title,
            'content': ' | '.join(text_content)
        }

    def scrape_page(self, url: str) -> Set[str]:
        """Scrape single page and return found links"""
        if url in self.visited_urls:
            return set()
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract page data
            self.extract_page_data(url, soup)
            self.visited_urls.add(url)
            
            # Find all links
            links = set()
            for link in soup.find_all('a', href=True):
                full_url = urllib.parse.urljoin(self.base_url, link['href'])
                if self.is_valid_url(full_url):
                    links.add(full_url)
            
            time.sleep(1)  # Rate limiting
            return links
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return set()

    def scrape_site(self):
        """Main scraping method"""
        to_visit = {self.base_url}
        
        while to_visit and len(self.visited_urls) < self.max_pages:
            current_url = to_visit.pop()
            new_links = self.scrape_page(current_url)
            to_visit.update(new_links - self.visited_urls)
            
            logging.info(f"Scraped: {current_url}")
            logging.info(f"Pages scraped: {len(self.visited_urls)}")

    def save_to_csv(self, filename: str = f'{dirname}/scraped_data.csv'):
        """Save scraped data to CSV"""
        df = pd.DataFrame.from_dict(self.data_dict, orient='index')
        df.to_csv(filename)
        logging.info(f"Data saved to {filename}")

def main():
    # Example usage
    base_url = "https://wiki.python.org/moin/PythonBooks"  # Replace with target website
    scraper = WebScraper(base_url, max_pages=500)
    scraper.scrape_site()
    scraper.save_to_csv()
    

if __name__ == "__main__":
    main()