"""
Author: Ian Young

Dense Passage Retrieval (DPR) module for document chunk retrieval.

This module provides functionality for creating a Dense Passage Retriever
and performing semantic document retrieval using pre-trained DPR models
from Facebook. It enables efficient chunk selection based on query similarity.
"""

from os.path import exists

import torch
from tqdm import tqdm


def embed_chunks(chunks, model, tokenizer, device):
    """
    Embeds document chunks using the context encoder.

    Args:
        chunks (list): List of document text chunks.
        model: The model used for generating text.
        tokenizer: The tokenizer for encoding the text.
        device (str): Device to use for compute.

    Returns:
        list: List of embeddings for the chunks.
    """
    embeddings = []

    for chunk in tqdm(chunks, desc="Embedding chunks", unit="chunk"):
        inputs = tokenizer(
            chunk["data"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=192,
        ).to(device)

        with torch.no_grad():
            outputs = model.encoder(**inputs)
            embedding = outputs.last_hidden_state.mean(
                dim=1
            )  # Aggregate embeddings
        embeddings.append(embedding)

    return torch.cat(embeddings)


def retrieve(query, chunks, model, tokenizer, device):
    """
    Retrieves the most relevant chunk for a given query.

    Args:
        query (str): The question to retrieve relevant chunks for.
        chunks (list): List of document text chunks.
        model: The model used for generating the text.
        tokenizer: The tokenizer for encoding the text.
        device (str): Device to use for compute.

    Returns:
        str: The most relevant chunk of text.
    """
    model.to(device)
    # Encode the query using the question encoder
    question_inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192,
    ).to(device)
    with torch.no_grad():
        question_embedding = model.encoder(**question_inputs)[0].mean(dim=1)

    # Get chunk embeddings
    if exists("chunk_embeddings.emb"):
        # Load pre-processed embeddings
        print("Loading pre-processed chunks.")
        chunk_embeddings = torch.load(
            "chunk_embeddings.emb", map_location=device, weights_only=True
        )
    else:
        # Encode the chunks
        chunk_embeddings = embed_chunks(chunks[::20], model, tokenizer, device)
        torch.save(chunk_embeddings, "chunk_embeddings.emb")

    # Calculate similarity between the query and each chunk (cosine similarity)
    similarities = []
    for chunk_embedding in chunk_embeddings:
        similarity = torch.cosine_similarity(
            question_embedding, chunk_embedding
        )
        similarities.append(similarity.item())

    # Find the chunk with the highest similarity
    best_chunk_idx = similarities.index(max(similarities))
    return chunks[best_chunk_idx]
