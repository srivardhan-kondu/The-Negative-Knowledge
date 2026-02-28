"""
Domain Configuration Manager for MindGap Explorer
Loads and manages domain-specific settings from config.yaml
"""

import yaml
from pathlib import Path

class DomainConfig:
    def __init__(self, config_path="config.yaml"):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load default domain
        self.current_domain = self.config.get('default_domain', 'mental_health')
        self.domain_data = None
        self.load_domain(self.current_domain)
    
    def load_domain(self, domain_name):
        """Switch to a different domain"""
        if domain_name not in self.config['domains']:
            available = list(self.config['domains'].keys())
            raise ValueError(f"Domain '{domain_name}' not found. Available: {available}")
        
        self.current_domain = domain_name
        self.domain_data = self.config['domains'][domain_name]
        print(f"✓ Loaded domain: {self.domain_data['name']}")
        return self.domain_data
    
    def get_search_terms(self):
        """Get search terms for current domain"""
        return self.domain_data['search_terms']
    
    def get_entity_categories(self):
        """Get entity categories and their keywords"""
        return self.domain_data['entity_categories']
    
    def get_domain_name(self):
        """Get human-readable domain name"""
        return self.domain_data['name']
    
    def get_domain_description(self):
        """Get domain description"""
        return self.domain_data['description']
    
    def classify_entity(self, entity_text):
        """Classify an entity into a category based on keywords"""
        entity_lower = entity_text.lower()
        
        for category, data in self.domain_data['entity_categories'].items():
            keywords = data['keywords']
            if any(keyword in entity_lower for keyword in keywords):
                return category
        
        return None
    
    def get_category_color(self, category):
        """Get color for a specific category"""
        if category in self.domain_data['entity_categories']:
            return self.domain_data['entity_categories'][category].get('color', '#cccccc')
        return '#cccccc'
    
    def list_available_domains(self):
        """List all available domains"""
        return list(self.config['domains'].keys())
    
    def get_api_settings(self):
        """Get API configuration"""
        return self.config.get('semantic_scholar', {})

# Usage example
if __name__ == "__main__":
    config = DomainConfig()
    
    print(f"\nCurrent Domain: {config.get_domain_name()}")
    print(f"Description: {config.get_domain_description()}")
    print(f"\nAvailable domains: {config.list_available_domains()}")
    print(f"\nSearch terms ({len(config.get_search_terms())} total):")
    for term in config.get_search_terms()[:3]:
        print(f"  - {term}")
    print("  ...")
    
    print(f"\nEntity categories:")
    for cat, data in config.get_entity_categories().items():
        print(f"  - {cat}: {len(data['keywords'])} keywords")
