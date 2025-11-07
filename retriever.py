"""TF-IDF based retrieval of relevant workflow examples."""

import json
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix


def load_examples(path: str) -> list[dict[str, Any]]:
    """Load example workflows from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    return examples


def to_text(example: dict[str, Any]) -> str:
    """Convert example to searchable text (title + description + config)."""
    title = example.get('title', '')
    description = example.get('description', '')
    config = example.get('config', {})
    config_str = json.dumps(config, separators=(',', ':'))
    return f"{title} {description} {config_str}"


def build_index(examples: list[dict[str, Any]]) -> tuple[TfidfVectorizer, csr_matrix, list[str]]:
    """Build TF-IDF index from examples. Returns (vectorizer, matrix, texts)."""
    if not examples:
        raise ValueError("Examples list cannot be empty")
    
    texts = [to_text(example) for example in examples]
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return vectorizer, tfidf_matrix, texts


def get_top_k(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    examples: list[dict[str, Any]],
    k: int = 3
) -> list[tuple[dict[str, Any], float]]:
    """Retrieve top-k most relevant examples using cosine similarity."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > len(examples):
        k = len(examples)
    
    query_vec = vectorizer.transform([query])
    similarity_scores = matrix.dot(query_vec.T).toarray().flatten()
    top_indices = np.argsort(similarity_scores)[::-1][:k]
    
    results = [
        (examples[idx], round(float(similarity_scores[idx]), 4))
        for idx in top_indices
    ]
    
    return results


if __name__ == "__main__":
    import os
    
    examples_path = os.path.join("data", "examples.json")
    print(f"Loading examples from {examples_path}...")
    examples = load_examples(examples_path)
    print(f"Loaded {len(examples)} examples\n")
    
    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix, texts = build_index(examples)
    print(f"Index built with vocabulary size: {len(vectorizer.vocabulary_)}\n")
    
    test_query = "notify if duration > 2 hours"
    print(f"Query: '{test_query}'")
    print("=" * 80)
    
    top_examples = get_top_k(test_query, vectorizer, tfidf_matrix, examples, k=3)
    
    print(f"\nTop {len(top_examples)} results:\n")
    for i, (example, score) in enumerate(top_examples, 1):
        print(f"Rank {i} (Score: {score}):")
        print(f"  ID: {example.get('id')}")
        print(f"  Title: {example.get('title')}")
        print(f"  Description: {example.get('description')}")
        print(f"  Config: {json.dumps(example.get('config'), indent=2)}")
        print()
