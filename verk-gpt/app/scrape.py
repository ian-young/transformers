import requests
from bs4 import BeautifulSoup


# Function to scrape content from a webpage
def scrape_website(urls, visited_urls):
    data = ""

    while urls:
        url = urls.pop(0)  # Get the next URL to scrape
        if url in visited_urls:
            print(f"Skipping {url}")
            continue  # Skip URLs we've already visited

        print(f"Scraping {url}...")
        visited_urls.add(url)  # Mark this URL as visited

        response = requests.get(url)  # Make the request to the URL
        soup = BeautifulSoup(response.content, "html.parser")

        # Get all paragraphs and headings
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        content = " ".join([para.get_text() for para in paragraphs])
        data += content + "\n\n"

        # Scrape links from this page and add to the list
        scrape_links(url, soup, urls, visited_urls)

    # Save the scraped data to a file
    with open("verkada_data.txt", "w", encoding="utf-8") as file:
        file.write(data)
    print("Data scraped and saved to verkada_data.txt")


def scrape_links(url, soup, urls, visited_urls):
    # Find all <a> tags with href attribute
    links = soup.find_all("a", href=True)

    for link in links:
        href = link.get("href")
        if href:
            # Skip fragment links (e.g., #content) or links with invalid schemes
            if href.startswith("#") or not href.startswith(("http", "https")):
                continue

            # Make relative URLs absolute
            full_url = f"{url}{href}" if href.startswith("/") else href

            # Check if the URL contains unwanted domains (google.com, facebook.com)
            if any(
                domain in full_url
                for domain in [
                    "google.com",
                    "facebook.com",
                    "instagram.com",
                    "twitter.com",
                    "youtube.com",
                    "linkedin.com",
                    "apple.com",
                ]
            ):
                # print(f"Skipping URL from unwanted domain: {full_url}")
                continue

            # Check if this URL has already been added to avoid duplicates
            if full_url not in urls:
                urls.append(full_url)  # Add the URL to the list
                # print(f"Found URL: {full_url}")

    # Scrape any new URLs found (avoid scraping the same URL again)
    for found_url in urls:
        # print(f"Scraping next: {found_url}")
        scrape_website([found_url], visited_urls)


if __name__ == "__main__":
    visited_urls = set()
    scrape_website(["https://help.verkada.com"], visited_urls)
