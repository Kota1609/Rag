import os
import requests
from urllib.parse import urlparse
from collections import defaultdict

def download_html(url, save_dir):
    try:
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        
        # Create a valid filename based on the URL
        parsed_url = urlparse(url)
        url_path = parsed_url.path.replace('/', '_')
        filename = f"{parsed_url.netloc}{url_path}.html"
        filename = filename.replace(":", "_").replace("?", "_").replace("#", "_")
        
        # Ensure the filename is unique in the directory to avoid overwrites
        base_filename = filename
        counter = 1
        while os.path.exists(os.path.join(save_dir, filename)):
            filename = f"{os.path.splitext(base_filename)[0]}_{counter}{os.path.splitext(base_filename)[1]}"
            counter += 1

        # Save the HTML content to a file
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        print(f"Saved {url} as {file_path}")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

def download_html_from_urls(file_path, save_dir):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    with open(file_path, "r") as file:
        urls = file.read().splitlines()
    
    for url in urls:
        if url:
            download_html(url, save_dir)

# Specify the input file and the directory to save the HTML files
input_file = "cs_filtered_links.txt"
output_dir = "cs_downloaded_html_pages"

# Download the HTML pages
download_html_from_urls(input_file, output_dir)

