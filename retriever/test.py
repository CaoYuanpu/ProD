"""Smoke test for a running retriever server: python retriever/test.py"""
from smolagents import WikipediaRetrieverTool

print(WikipediaRetrieverTool().forward("Berlinda Tolbert CBS sitcom"))
