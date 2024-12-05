import threading
import math
import re
from os import cpu_count
from concurrent.futures import ThreadPoolExecutor

import fitz
import requests
from bs4 import BeautifulSoup

# Define a headers dictionary with a common User-Agent string
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Create and clear out the file
with open("verkada_data.txt", "w", encoding="utf-8") as v_file:
    v_file.write("")


def chunk_urls(urls, num_chunks):
    # Calculate the chunk size (rounded up to handle uneven chunks)
    chunk_size = math.ceil(len(urls) / num_chunks)
    return [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]


def scrape_website(urls, visited_urls, lock):
    max_workers = min(cpu_count() * 2, len(urls))
    url_chunks = chunk_urls(urls, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk in url_chunks:
            executor.submit(scrape_urls, chunk, visited_urls, lock)


# Function to scrape content from a webpage
def scrape_urls(urls, visited_urls, lock):
    while urls:
        url = urls.pop(0)  # Get the next URL to scrape
        with lock:
            if url in visited_urls:
                continue  # Skip URLs we've already visited

            visited_urls.add(url)  # Mark this URL as visited

        print(f"Scraping {url}...")
        data = ""
        response = requests.get(
            url, headers=headers, timeout=5
        )  # Make the request to the URL
        soup = BeautifulSoup(response.content, "html.parser")

        if "application/pdf" in response.headers.get("Content-Type", ""):
            print(f"Extracting text from PDF at {url}...")
            pdf_text = extract_text_from_pdf(response.content)
            data += pdf_text + "\n\n"
        elif "text/html" in response.headers.get("Content-Type", ""):

            # Get all paragraphs and headings
            paragraphs = soup.find_all(
                ["p", "h1", "h2", "h3", "h4", "h5", "h6"]
            )
            content = " ".join(
                [
                    para.get_text().strip()
                    for para in paragraphs
                    if "Sorry, Internet Explorer is not supported"
                    not in para.get_text()
                ]
            )
            data += content + "\n\n"
        else:
            print(f"Skipping non-HTML content from {url}")

        # Save the scraped data to a file
        with lock:
            with open("verkada_data.txt", "a", encoding="utf-8") as file:
                file.write(data)

        with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
            # Scrape links from this page and add to the list
            executor.submit(scrape_links, url, soup, urls, visited_urls, lock)


def extract_text_from_pdf(pdf_content):
    """Extract text from a PDF file using PyMuPDF (fitz), with all content on one line."""
    text = ""
    with fitz.open(stream=pdf_content, filetype="pdf") as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            page_text = page.get_text(
                "text"
            )  # Extracts text using simple text extraction
            text += page_text

    # Remove newlines, carriage returns, and extra spaces
    text = " ".join(text.split())
    return text


def scrape_links(url, soup, urls, visited_urls, lock):
    # Find all <a> tags with href attribute
    links = soup.find_all("a", href=True)

    for link in links:
        if href := link.get("href"):
            # Skip fragment links (e.g., #content) or links with invalid schemes
            if href.startswith("#") or not href.startswith(("http", "https")):
                continue

            # Make relative URLs absolute
            full_url = f"{url}{href}" if href.startswith("/") else href

            # Check if the URL contains "verkada"
            if not re.search(r"verkada", full_url, re.IGNORECASE) or any(
                domain in full_url
                for domain in [
                    "linkdin.com",
                    "github.com",
                    "verkada.intercom-attachements-7.com",
                ]
            ):
                continue  # Skip the URL if it does not

            # Check if this URL has already been added to avoid duplicates
            if full_url not in urls:
                urls.append(full_url)  # Add the URL to the list

    # Scrape any new URLs found (avoid scraping the same URL again)
    for found_url in urls:
        scrape_website([found_url], visited_urls, lock)


if __name__ == "__main__":
    processed_urls = set()
    thread_lock = threading.Lock()
    try:
        scrape_website(
            ["https://docs.verkada.com"], processed_urls, thread_lock
        )
    except KeyboardInterrupt:
        print("Exiting...")
