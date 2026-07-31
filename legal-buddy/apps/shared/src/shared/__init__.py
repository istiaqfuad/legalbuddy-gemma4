"""Shared primitives for the legal RAG pipeline: embedding, qdrant, chunking.

Single source of truth used by both the API (query side) and ingestion (passage
side) so the embedding model, e5 prefixes, vector params, and chunking can never
drift out of sync.
"""
