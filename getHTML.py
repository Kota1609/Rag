

import requests
from urls import geturls
from urllib.parse import urlparse, unquote

# List of URLs to download
urls = geturls()

# Common browser User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
}

def generate_filename(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    # Remove leading slash, replace remaining slashes with dashes, decode URL-encoded characters
    normalized_path = unquote(path[1:].replace('/', '-'))
    # Remove file extension if any, typically ".html"
    if normalized_path.endswith('.html'):
        normalized_path = normalized_path[:-5]
    return normalized_path if normalized_path else "index"

# Loop through each URL and download the HTML content
for url in urls:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        filename = generate_filename(url) + ".html"  # Ensures all files end with .html
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {url}: Status code {response.status_code}")
