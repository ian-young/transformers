import requests
from bs4 import BeautifulSoup


# Function to scrape content from a webpage
def scrape_website(urls):
    data = ""
    for url in urls:  # Iterate over the list of URLs
        print(f"Scraping {url}...")
        response = requests.get(url)  # Make the request to each URL
        soup = BeautifulSoup(response.content, "html.parser")

        # Get all paragraphs and headings
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        content = " ".join([para.get_text() for para in paragraphs])
        data += content + "\n\n"

    # Save the scraped data to a file
    with open("verkada_data.txt", "w", encoding="utf-8") as file:
        file.write(data)
    print("Data scraped and saved to verkada_data.txt")
