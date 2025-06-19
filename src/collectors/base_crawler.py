import asyncio
import aiohttp
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import logging
from datetime import datetime
from typing import List, Dict, Optional
import time
from urllib.parse import urljoin, urlparse
from config.settings import settings

class BaseCrawler(ABC):
    def __init__(self, source_config: Dict):
        self.name = source_config['name']
        self.url = source_config['url']
        self.type = source_config['type']
        self.selectors = source_config.get('selectors', {})
        self.crawl_frequency = source_config.get('crawl_frequency', 'daily')
        self.logger = logging.getLogger(f"crawler.{self.name}")
        self.session_timeout = aiohttp.ClientTimeout(total=settings.CRAWL_TIMEOUT)
        
    async def fetch_page(self, url: str, retries: int = 0) -> Optional[str]:
        """Fetch page content with retry logic"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        self.logger.debug(f"Successfully fetched {url}")
                        return content
                    elif response.status == 429 and retries < settings.MAX_RETRIES:
                        # Rate limited, wait and retry
                        wait_time = 2 ** retries
                        self.logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        return await self.fetch_page(url, retries + 1)
                    else:
                        self.logger.error(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
        except asyncio.TimeoutError:
            if retries < settings.MAX_RETRIES:
                self.logger.warning(f"Timeout fetching {url}, retrying...")
                return await self.fetch_page(url, retries + 1)
            else:
                self.logger.error(f"Timeout fetching {url} after {retries} retries")
                return None
        except Exception as e:
            if retries < settings.MAX_RETRIES:
                self.logger.warning(f"Error fetching {url}: {str(e)}, retrying...")
                await asyncio.sleep(1)
                return await self.fetch_page(url, retries + 1)
            else:
                self.logger.error(f"Failed to fetch {url} after {retries} retries: {str(e)}")
                return None
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content"""
        return BeautifulSoup(html, 'html.parser')
    
    def extract_text(self, element) -> str:
        """Extract and clean text from HTML element"""
        if not element:
            return ""
        text = element.get_text(strip=True)
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        return ' '.join(text.split()).strip()
    
    def make_absolute_url(self, relative_url: str) -> str:
        """Convert relative URL to absolute"""
        if not relative_url:
            return ""
        return urljoin(self.url, relative_url)
    
    def is_valid_article_url(self, url: str) -> bool:
        """Check if URL looks like a valid article"""
        if not url:
            return False
        
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Skip common non-article paths
        skip_patterns = [
            '/category/', '/tag/', '/author/', '/search/',
            '/archive/', '/page/', '.pdf', '.jpg', '.png',
            '.gif', '/contact', '/about', '/privacy'
        ]
        
        for pattern in skip_patterns:
            if pattern in path:
                return False
        
        return True
    
    @abstractmethod
    async def extract_signals(self, html: str) -> List[Dict]:
        """Extract signals from HTML content - must be implemented by subclasses"""
        pass
    
    async def extract_article_links(self, html: str) -> List[str]:
        """Extract article links from main page"""
        soup = self.parse_html(html)
        links = []
        
        if 'articles' in self.selectors and 'link' in self.selectors:
            articles = soup.select(self.selectors['articles'])
            for article in articles:
                link_element = article.select_one(self.selectors['link'])
                if link_element and link_element.get('href'):
                    url = self.make_absolute_url(link_element['href'])
                    if self.is_valid_article_url(url):
                        links.append(url)
        
        self.logger.info(f"Found {len(links)} article links")
        return links[:50]  # Limit to prevent overloading
    
    async def extract_article_content(self, url: str) -> Optional[Dict]:
        """Extract content from individual article"""
        html = await self.fetch_page(url)
        if not html:
            return None
        
        soup = self.parse_html(html)
        
        # Try to extract title
        title = ""
        title_selectors = ['h1', 'title', '.article-title', '.post-title']
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                title = self.extract_text(title_element)
                break
        
        # Try to extract content
        content = ""
        content_selectors = [
            '.article-content', '.post-content', '.entry-content',
            'article', '.content', 'main'
        ]
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content = self.extract_text(content_element)
                if len(content) > 100:  # Only use if substantial content
                    break
        
        if not title and not content:
            return None
        
        return {
            'title': title,
            'content': content,
            'url': url,
            'raw_html': html,
            'extracted_at': datetime.utcnow()
        }
    
    async def crawl(self) -> List[Dict]:
        """Main crawl method"""
        start_time = time.time()
        self.logger.info(f"Starting crawl for {self.name}")
        
        try:
            # Fetch main page
            main_html = await self.fetch_page(self.url)
            if not main_html:
                self.logger.error(f"Failed to fetch main page for {self.name}")
                return []
            
            # Extract signals from main page first
            signals = await self.extract_signals(main_html)
            
            # Also try to get individual articles
            article_links = await self.extract_article_links(main_html)
            
            # Process a subset of articles to avoid overloading
            for url in article_links[:10]:
                article_data = await self.extract_article_content(url)
                if article_data:
                    article_signals = await self.extract_signals(article_data['raw_html'])
                    # Add article-specific metadata
                    for signal in article_signals:
                        signal.update({
                            'article_title': article_data['title'],
                            'article_url': article_data['url']
                        })
                    signals.extend(article_signals)
                
                # Small delay to be respectful
                await asyncio.sleep(0.5)
            
            processing_time = time.time() - start_time
            self.logger.info(
                f"Crawl completed for {self.name}: "
                f"{len(signals)} signals in {processing_time:.2f}s"
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error during crawl for {self.name}: {str(e)}")
            return []

class GenericCrawler(BaseCrawler):
    """Generic crawler that uses CSS selectors from config"""
    
    async def extract_signals(self, html: str) -> List[Dict]:
        """Extract signals using CSS selectors"""
        soup = self.parse_html(html)
        signals = []
        
        if self.selectors and 'articles' in self.selectors:
            articles = soup.select(self.selectors['articles'])
            
            for article in articles:
                title = ""
                content = ""
                link = ""
                
                # Extract title
                if 'title' in self.selectors:
                    title_element = article.select_one(self.selectors['title'])
                    title = self.extract_text(title_element)
                
                # Extract summary/content
                if 'summary' in self.selectors:
                    summary_element = article.select_one(self.selectors['summary'])
                    content = self.extract_text(summary_element)
                
                # Extract link
                if 'link' in self.selectors:
                    link_element = article.select_one(self.selectors['link'])
                    if link_element and link_element.get('href'):
                        link = self.make_absolute_url(link_element['href'])
                
                if title or content:
                    signals.append({
                        'title': title,
                        'content': content or title,  # Use title as content if no summary
                        'url': link,
                        'source_name': self.name,
                        'source_type': self.type,
                        'extracted_at': datetime.utcnow()
                    })
        
        return signals