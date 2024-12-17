"""
Author: Ian Young

Dense Passage Retrieval (DPR) module for document chunk retrieval.

This module provides functionality for creating a Dense Passage Retriever
and performing semantic document retrieval using pre-trained DPR models
from Facebook. It enables efficient chunk selection based on query similarity.
"""

import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder


def create_retriever(device):
    """
    Creates a Dense Passage Retriever (DPR) for chunk retrieval.

    Args:
        device (str): Device to use for compute.

    Returns:
        context_encoder, question_encoder: DPR encoder models.
    """
    # Load the pre-trained DPR models
    context_encoder = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    ).to(device)
    question_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    ).to(device)
    return context_encoder, question_encoder


def embed_chunks(chunks, context_encoder, tokenizer, device):
    """
    Embeds document chunks using the context encoder.

    Args:
        chunks (list): List of document text chunks.
        context_encoder: The DPR context encoder model.
        tokenizer: The tokenizer for encoding the text.
        device (str): Device to use for compute.

    Returns:
        list: List of embeddings for the chunks.
    """
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            embedding = context_encoder(
                **inputs
            ).pooler_output  # Get the chunk embedding
        embeddings.append(embedding)

    # Stack embeddings into a single tensor for efficient computation.
    # Tensor stacks utilize vectorized operations.
    return torch.stack(embeddings)


def retrieve(
    query, chunks, question_encoder, context_encoder, tokenizer, device
):
    """
    Retrieves the most relevant chunk for a given query.

    Args:
        query (str): The question to retrieve relevant chunks for.
        chunks (list): List of document text chunks.
        question_encoder: The DPR question encoder model.
        context_encoder: The DPR context encoder model.
        tokenizer: The tokenizer for encoding the text.
        device (str): Device to use for compute.

    Returns:
        str: The most relevant chunk of text.
    """
    # Encode the query using the question encoder
    question_inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        question_embedding = question_encoder(
            **question_inputs
        ).pooler_output  # Get the query embedding

    # Encode the chunks
    chunk_embeddings = embed_chunks(chunks, context_encoder, tokenizer, device)

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
