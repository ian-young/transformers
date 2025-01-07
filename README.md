# transformers

This repository contains a transformer that will crawl through websites, carve out the data, format the data into SQuAD format, then train into a Large T5 Q&A model to answer specialized questions.

The purpose of this is to create a custom, specalized model that will be saved and accessed later using a Golang script that allows for easy terminal integration.

The flow of the script is as follows:

```mermaid
    flowchart TB
    subgraph retriever.py
        retrieve
        embed_chunks
    end
    subgraph tune.py
        set_torch_device
        Callback[[PauseTrainingCallback]]
        scrape_and_save
        compute_metrics
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
        generate_squad_format_with_checkpoint
        chunk_text
        preprocess_custom_data
        process_chunks
        prepare_squad_data
        tokenize_data_item

    end

    main.py ==> scrape_and_save ==> set_torch_device ==> train_model ==> retrieve

    scrape_and_save -.-> scrape_website
    scrape_website --> chunk_urls --> scrape_urls
    scrape_urls --PDF--> extract_text_from_pdf
    scrape_urls --Webapge--> scrape_links

    train_model -.-> preprocess_custom_data -.-> compute_metrics
    train_model --- Callback

    preprocess_custom_data --> chunk_text --> generate_squad_format_with_checkpoint --> process_chunks --> prepare_squad_data --> tokenize_data_item
    prepare_squad_data --> tokenize_data_item

    retrieve --> embed_chunks
```
