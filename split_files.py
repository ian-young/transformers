"""Split Verkada data to individual files"""

import json
import re
from tqdm import tqdm

# Load the scraped data from a file
with open("verk-gpt/verkada_data_backup.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# Split and chunk the text into individual documents
documents = [json.loads(doc) for doc in text_data.split("\n\n") if doc.strip()]

# Create a progress bar to display the progress of splitting the data
progress_bar = tqdm(
    total=len(documents),
    desc="Splitting documents",
    unit="Document",
    colour="green",
)

# Iterate over each document and write it to a new file with its URL as the filename
for line in documents:
    # Extract the URL from the current document
    url = line["url"]

    # Sanitize the URL: Replace invalid filename characters with underscores
    url = re.sub(r"https?://", "", url)  # Remove 'http://' or 'https://'
    # url = url.split('/')[0] + url.split('/')[1]  # Keep only the domain part (before the first '/')
    url = re.sub(r'[\/:*?"<>|]', "_", url)
    url = url[:50]

    # Write the text of the current document to a new file with its sanitized URL as the filename
    with open(
        f"/Users/ianyoung/Documents/verkada-files/{url}.txt",
        "w",
        encoding="utf-8",
    ) as out_file:
        out_file.write(str(line["text"]))

    # Update the progress bar after writing each document to a new file
    progress_bar.update(1)

# Close the progress bar when all documents have been processed
progress_bar.close()
