"""
Author: Ian Young

This module provides functionality for web scraping, including the ability
to extract content from websites and handle various content types such as
HTML and PDF.

It includes functions to chunk URLs, scrape websites concurrently, extract
text from PDF documents, and process links found on scraped pages. The
module utilizes threading to enhance performance and ensure thread safety
when accessing shared resources.

Functions:
    chunk_urls: Splits a list of URLs into a specified number of chunks.
    scrape_website: Initiates the scraping of multiple websites
        concurrently.
    scrape_urls: Scrapes content from a list of URLs and saves the data
        to a file.
    extract_text_from_pdf: Extracts text from a PDF document provided
        as binary content.
    scrape_links: Extracts and processes links from a given webpage's
        HTML content.

Usage:
    The module can be run as a standalone script to scrape a predefined
    list of URLs. It will manage the scraping process, handle threading,
    and save the scraped data to a file.
"""

import threading
import math
import re
from os import cpu_count
from json import dumps
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
    """
    Splits a list of URLs into a specified number of chunks.

    This function calculates the appropriate size for each chunk and
    divides the input list of URLs accordingly. It ensures that all URLs
    are included, even if the total number of URLs does not evenly divide
    by the number of chunks.

    Args:
        urls (list): A list of URLs to be chunked.
        num_chunks (int): The desired number of chunks to create.

    Returns:
        list: A list of lists, where each sublist contains a chunk of URLs.

    Examples:
        chunks = chunk_urls(["http://example.com", "http://example.org"], 2)
    """
    # Calculate the chunk size (rounded up to handle uneven chunks)
    chunk_size = math.ceil(len(urls) / num_chunks)
    return [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]


def scrape_website(urls, visited_urls, lock):
    """
    Initiates the scraping of multiple websites concurrently.

    This function divides a list of URLs into chunks and uses a thread
    pool to scrape each chunk in parallel. It ensures that the scraping
    process is thread-safe by utilizing a lock to manage access to
    shared resources.

    Args:
        urls (list): A list of URLs to be scraped.
        visited_urls (set): A set to keep track of URLs that have already
            been visited.
        lock (Lock): A threading lock to synchronize access to shared
            resources.

    Returns:
        None

    Examples:
        scrape_website(["http://example.com", "http://example.org"], visited_urls, lock)
    """
    max_workers = min(cpu_count() * 2, len(urls))
    url_chunks = chunk_urls(urls, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk in url_chunks:
            executor.submit(scrape_urls, chunk, visited_urls, lock)


# Function to scrape content from a webpage
def scrape_urls(urls, visited_urls, lock):
    """
    Scrapes content from a list of URLs and saves the data to a file.

    This function processes each URL by checking if it has already been
    visited, scraping its content, and extracting relevant information
    based on the content type. It handles both HTML and PDF content,
    ensuring that the scraped data is stored safely in a file while
    managing concurrent access with a lock.

    Args:
        urls (list): A list of URLs to be scraped.
        visited_urls (set): A set to track URLs that have already been
            visited.
        lock (Lock): A threading lock to synchronize access to shared
            resources.

    Returns:
        None

    Examples:
        scrape_urls(urls, visited_urls, lock)
    """
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
            data = extract_text_from_pdf(response.content)
        elif "text/html" in response.headers.get("Content-Type", ""):

            # Get all paragraphs and headings
            paragraphs = soup.find_all(
                ["p", "h1", "h2", "h3", "h4", "h5", "h6"]
            )
            data = "\n".join(
                [
                    para.get_text().strip()
                    for para in paragraphs
                    if "Sorry, Internet Explorer is not supported"
                    not in para.get_text()
                ]
            )
        else:
            print(f"Skipping non-HTML content from {url}")

        data_list = [{"url": url, "text": data}]
        # Save the scraped data to a file
        with lock:
            with open("verkada_data.txt", "a", encoding="utf-8") as file:
                print(data_list)
                file.write(dumps(data_list, ensure_ascii=False) + "\n\n")

        with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
            # Scrape links from this page and add to the list
            executor.submit(scrape_links, url, soup, urls, visited_urls, lock)


def extract_text_from_pdf(pdf_content):
    """
    Extracts text from a PDF document provided as binary content.

    This function reads the PDF content and retrieves the text from
    each page, concatenating it into a single string. It also cleans
    up the extracted text by removing unnecessary whitespace and newlines
    for better readability.

    Args:
        pdf_content (bytes): The binary content of the PDF file.

    Returns:
        str: The extracted text from the PDF.

    Examples:
        text = extract_text_from_pdf(pdf_binary_content)
    """
    text = ""
    with fitz.open(stream=pdf_content, filetype="pdf") as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            page_text = page.get_text(
                "text"
            )  # Extracts text using simple text extraction
            text += page_text

    # Remove newlines, carriage returns, and extra spaces
    # text = " ".join(text.split())
    return text


def scrape_links(url, soup, urls, visited_urls, lock):
    """
    Extracts and processes links from a given webpage's HTML content.

    This function identifies all anchor tags in the provided HTML soup,
    filters the links based on specific criteria, and adds valid links
    to a list for further scraping. It ensures that only relevant and
    unique URLs are considered, avoiding duplicates and links that do
    not match the desired domain.

    Args:
        url (str): The base URL of the webpage being scraped.
        soup (BeautifulSoup): The BeautifulSoup object containing the
            parsed HTML of the webpage.
        urls (list): A list to store newly found URLs for scraping.
        visited_urls (set): A set to track URLs that have already been
            visited.
        lock (Lock): A threading lock to synchronize access to shared
            resources.

    Returns:
        None

    Examples:
        scrape_links("http://example.com", soup, urls, visited_urls, lock)
    """
    print("Hunting for more links")
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
            if (
                not re.search(r"verkada", full_url, re.IGNORECASE)
                or any(
                    domain in full_url
                    for domain in [
                        "linkdin.com",
                        "github.com",
                        "verkada.intercom-attachements-7.com",
                    ]
                )
                or re.search(r"verkada.com/ja", full_url, re.IGNORECASE)
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
