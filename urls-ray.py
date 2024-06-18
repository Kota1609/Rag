import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import ray

ray.init()

@ray.remote
def fetch_url(url, base_domain, session_headers):
    headers = session_headers.copy()
    try:
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.get(url)
            response.raise_for_status()
            try:
                soup = BeautifulSoup(response.content, 'lxml')
            except Exception as e:
                print(f"Error with lxml parser for {url}: {e}. Trying html.parser.")
                soup = BeautifulSoup(response.content, 'html.parser')

            links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if is_valid_url(full_url) and is_same_domain(full_url, base_domain):
                    links.add(full_url)
            return list(links)
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return []
    except Exception as e:
        print(f"Error parsing content from {url}: {e}")
        return []

def is_same_domain(url, base_domain):
    return urlparse(url).netloc == base_domain

def is_valid_url(url):
    return not any(url.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.docx'])

def scrape_site(base_url, max_links=2000):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    base_domain = urlparse(base_url).netloc
    visited = set()
    to_visit = [base_url]

    while to_visit and len(visited) < max_links:
        current_urls = to_visit[:50]  # Fetch URLs in batches
        to_visit = to_visit[50:]

        # Start Ray tasks
        futures = [fetch_url.remote(url, base_domain, headers) for url in current_urls]
        results = ray.get(futures)

        for links in results:
            for link in links:
                if link not in visited and len(visited) < max_links:
                    visited.add(link)
                    to_visit.append(link)

        print(f"Visited: {len(visited)} links")

    return visited

# Example usage
# url = 'www.meconlimited.co.in'
# print(f"Starting scraping for: {url}")
url = "https://computerscience.engineering.unt.edu/"
found_links = scrape_site(url, 200000)
print(f"Found links for {url}: {found_links}")

# Filter links that do not end with '#main'
filtered_urls = [url for url in found_links if not url.endswith("#main")]

def geturls():
    return filtered_urls

# Get the filtered URLs
urls_to_use = geturls()
#write_links_to_file("found_links.txt", urls_to_use)
#print(f"Filtered URLs: {urls_to_use}")
def write_links_to_file(filename, links):
    with open(filename, 'a') as file:
        for link in links:
            file.write(link + "\n")

ray.shutdown()
write_links_to_file("cs_found_links.txt", urls_to_use)
