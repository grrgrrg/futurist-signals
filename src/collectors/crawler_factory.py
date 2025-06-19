import yaml
from typing import List, Dict
from .base_crawler import GenericCrawler
import logging

logger = logging.getLogger(__name__)

class CrawlerFactory:
    """Factory for creating crawlers from configuration"""
    
    @staticmethod
    def load_sources(config_path: str = "config/sources.yaml") -> Dict:
        """Load source configurations from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load sources config: {str(e)}")
            return {}
    
    @staticmethod
    def create_crawler(source_config: Dict) -> GenericCrawler:
        """Create a crawler instance for a source"""
        return GenericCrawler(source_config)
    
    @staticmethod
    def create_all_crawlers() -> List[GenericCrawler]:
        """Create crawlers for all configured sources"""
        sources_config = CrawlerFactory.load_sources()
        crawlers = []
        
        for tier_name, sources in sources_config.get('sources', {}).items():
            for source_config in sources:
                try:
                    crawler = CrawlerFactory.create_crawler(source_config)
                    crawlers.append(crawler)
                    logger.info(f"Created crawler for {source_config['name']}")
                except Exception as e:
                    logger.error(f"Failed to create crawler for {source_config.get('name', 'unknown')}: {str(e)}")
        
        return crawlers

async def crawl_all_sources() -> List[Dict]:
    """Crawl all configured sources and return raw signals"""
    crawlers = CrawlerFactory.create_all_crawlers()
    all_signals = []
    
    for crawler in crawlers:
        try:
            signals = await crawler.crawl()
            all_signals.extend(signals)
        except Exception as e:
            logger.error(f"Error crawling {crawler.name}: {str(e)}")
    
    logger.info(f"Total raw signals collected: {len(all_signals)}")
    return all_signals