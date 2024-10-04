# utils.py
import re
import tldextract

def preprocess_url(url):
    """
    Preprocesses the given URL by performing various steps:
    1. Remove protocol (http, https)
    2. Extract domain and subdomain
    3. Remove query parameters, anchors, etc.
    """
    # Remove protocol (http://, https://) and www.
    url = re.sub(r"https?://(www\.)?", "", url)
    
    # Extract domain parts using tldextract
    extracted_url = tldextract.extract(url)
    domain = extracted_url.domain
    subdomain = extracted_url.subdomain
    suffix = extracted_url.suffix
    
    # Join domain parts (subdomain.domain.suffix)
    cleaned_url = f"{subdomain}.{domain}.{suffix}" if subdomain else f"{domain}.{suffix}"
    
    # Remove any remaining path, query parameters, anchors, etc.
    cleaned_url = re.sub(r"/.*", "", cleaned_url)
    
    return cleaned_url
