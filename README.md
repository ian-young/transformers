# Transformer Revamp

This repository contains a transformer that will crawl through websites, carve out the data, format the data into a JSONL file for [MLX](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data), then train into a base model of your choice to create your own custom model! Please use the custom training to adjust behavioral patterns, speech patterns and add non-volitile knowledge. It is highly recommended to use RAG for volitile knowledge/information that updates frequently.

The purpose of this is to create a custom, specalized model that can be accessed through [open-webui](https://github.com/open-webui/open-webui).

The flow of the script has been greatly simplified:

```mermaid
    flowchart LR
    subgraph tune.py
        scrape_and_save
        train_model
    end
    subgraph scrape.py
        chunk_urls
        scrape_website
        scrape_urls
        extract_text_from_pdf
        scrape_links
    end
    subgraph preprocess_data.py
        generate_qa_with_checkpoint
        chunk_text
        preprocess_custom_data
        process_chunks
        replace_unicode
        generate_qa_entry

    end

    main.py ==> scrape_and_save ==> train_model

    scrape_and_save -.-> scrape_website
    scrape_website --> chunk_urls --> scrape_urls
    scrape_urls --PDF--> extract_text_from_pdf
    scrape_urls --Webpage--> scrape_links

    train_model -.-> preprocess_custom_data

    preprocess_custom_data --> chunk_text --> generate_qa_with_checkpoint --> process_chunks --> generate_qa_entry --> replace_unicode
```
