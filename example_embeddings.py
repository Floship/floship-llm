"""
Example usage of Heroku Inference API embeddings with floship-llm.

This demonstrates the /v1/embeddings endpoint support.
See: https://devcenter.heroku.com/articles/heroku-inference-api-v1-embeddings
"""

import importlib.util
import os
import sys

from floship_llm import LLM

# Setup: Make sure environment variables are set
# export INFERENCE_URL="https://us.inference.heroku.com"
# export INFERENCE_MODEL_ID="cohere-embed-multilingual"
# export INFERENCE_KEY="your-heroku-api-key"

# Or use Heroku CLI:
# eval $(heroku config -a $APP_NAME --shell | grep '^INFERENCE_' | sed 's/^/export /' | tee >(cat >&2))


def example_basic_embedding():
    """Example 1: Basic single text embedding."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Single Text Embedding")
    print("=" * 60)

    llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
    )

    text = "Hello, I am a long string (document) and I want to be turned into a searchable embedding vector!"
    embedding = llm.embed(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Last 5 values: {embedding[-5:]}")


def example_multiple_embeddings():
    """Example 2: Generate embeddings for multiple texts at once."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Text Embeddings (Batch)")
    print("=" * 60)

    llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
    )

    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
    ]

    embeddings = llm.embed(texts)

    print(f"Generated {len(embeddings)} embeddings")
    for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
        print(f"\nText {idx + 1}: {text}")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 3 values: {embedding[:3]}")


def example_with_full_response():
    """Example 3: Get full API response with metadata."""
    print("\n" + "=" * 60)
    print("Example 3: Full Response with Metadata")
    print("=" * 60)

    llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
    )

    text = "This is a sample document for embedding."
    response = llm.embed(text, return_full_response=True)

    print(f"Text: {text}")
    print("\nFull Response Structure:")
    print(f"  Object type: {response['object']}")
    print(f"  Model used: {response['model']}")
    print(f"  Number of embeddings: {len(response['data'])}")

    if response["usage"]:
        print("\nToken Usage:")
        print(f"  Prompt tokens: {response['usage']['prompt_tokens']}")
        print(f"  Total tokens: {response['usage']['total_tokens']}")

    # Access the actual embedding
    embedding = response["data"][0]["embedding"]
    print("\nEmbedding Details:")
    print(f"  Dimension: {len(embedding)}")
    print(f"  Index: {response['data'][0]['index']}")


def example_search_document():
    """Example 4: Generate embeddings optimized for search (documents)."""
    print("\n" + "=" * 60)
    print("Example 4: Search Document Embeddings")
    print("=" * 60)

    llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
        input_type="search_document",  # Optimize for document indexing
    )

    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "JavaScript is the programming language of the web.",
        "Rust is a systems programming language focused on safety and performance.",
    ]

    embeddings = llm.embed(documents)

    print(f"Generated {len(embeddings)} document embeddings for search indexing")
    print("These embeddings are optimized for being searched against queries.")
    for idx, doc in enumerate(documents):
        print(f"\nDocument {idx + 1}: {doc[:60]}...")
        print(f"  Embedding dimension: {len(embeddings[idx])}")


def example_search_query():
    """Example 5: Generate embeddings optimized for search queries."""
    print("\n" + "=" * 60)
    print("Example 5: Search Query Embeddings")
    print("=" * 60)

    llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
        input_type="search_query",  # Optimize for search queries
    )

    queries = [
        "What is Python programming?",
        "How does JavaScript work?",
        "Tell me about Rust language",
    ]

    embeddings = llm.embed(queries)

    print(f"Generated {len(embeddings)} query embeddings")
    print("These embeddings are optimized for searching against document embeddings.")
    for idx, query in enumerate(queries):
        print(f"\nQuery {idx + 1}: {query}")
        print(f"  Embedding dimension: {len(embeddings[idx])}")


def example_classification():
    """Example 6: Generate embeddings for classification tasks."""
    print("\n" + "=" * 60)
    print("Example 6: Classification Embeddings")
    print("=" * 60)

    llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
        input_type="classification",  # Optimize for classification
    )

    texts = [
        "This product is amazing! I love it.",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special.",
    ]

    embeddings = llm.embed(texts)

    print(f"Generated {len(embeddings)} embeddings for classification")
    print("These embeddings can be used to train a classifier.")
    for idx, text in enumerate(texts):
        print(f"\nText {idx + 1}: {text}")
        print(f"  Embedding dimension: {len(embeddings[idx])}")


def example_clustering():
    """Example 7: Generate embeddings for clustering tasks."""
    print("\n" + "=" * 60)
    print("Example 7: Clustering Embeddings")
    print("=" * 60)

    llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
        input_type="clustering",  # Optimize for clustering
    )

    texts = [
        "Apple announces new iPhone model",
        "Google releases Android update",
        "Stock market reaches new high",
        "Economic indicators show growth",
        "Microsoft launches new product",
    ]

    embeddings = llm.embed(texts)

    print(f"Generated {len(embeddings)} embeddings for clustering")
    print("These embeddings can be used to group similar items together.")
    for idx, text in enumerate(texts):
        print(f"\nText {idx + 1}: {text}")
        print(f"  Embedding dimension: {len(embeddings[idx])}")


def example_similarity_search():
    """Example 8: Practical similarity search using embeddings."""
    print("\n" + "=" * 60)
    print("Example 8: Similarity Search Demo")
    print("=" * 60)

    import numpy as np

    # Create LLM instances for documents and queries
    doc_llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
        input_type="search_document",
    )

    query_llm = LLM(
        type="embedding",
        model=os.environ.get("INFERENCE_MODEL_ID", "cohere-embed-multilingual"),
        input_type="search_query",
    )

    # Create document embeddings
    documents = [
        "Paris is the capital of France and known for the Eiffel Tower.",
        "Tokyo is the capital of Japan and a major technology hub.",
        "New York is a major city in the United States.",
        "Machine learning is transforming artificial intelligence.",
        "Deep learning uses neural networks for complex tasks.",
    ]

    print("Indexing documents...")
    doc_embeddings = doc_llm.embed(documents)

    # Search with a query
    query = "What is the capital of France?"
    print(f"\nQuery: {query}")

    query_embedding = query_llm.embed(query)

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nSearching for most similar documents...")
    similarities = [
        (idx, cosine_similarity(query_embedding, doc_emb))
        for idx, doc_emb in enumerate(doc_embeddings)
    ]

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 Results:")
    for rank, (idx, score) in enumerate(similarities[:3], 1):
        print(f"\n{rank}. Document {idx + 1} (similarity: {score:.4f})")
        print(f"   {documents[idx]}")


if __name__ == "__main__":
    # Verify environment variables
    required_vars = ["INFERENCE_URL", "INFERENCE_MODEL_ID", "INFERENCE_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print("ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set them using:")
        print("  export INFERENCE_URL='https://us.inference.heroku.com'")
        print("  export INFERENCE_MODEL_ID='cohere-embed-multilingual'")
        print("  export INFERENCE_KEY='your-api-key'")
        print("\nOr use Heroku CLI:")
        print(
            "  eval $(heroku config -a $APP_NAME --shell | grep '^INFERENCE_' | sed 's/^/export /' | tee >(cat >&2))"
        )
        sys.exit(1)

    print("=" * 60)
    print("Heroku Inference API - Embeddings Examples")
    print("=" * 60)

    # Run all examples
    try:
        example_basic_embedding()
        example_multiple_embeddings()
        example_with_full_response()
        example_search_document()
        example_search_query()
        example_classification()
        example_clustering()

        # Check if numpy is available for similarity search
        if importlib.util.find_spec("numpy") is not None:
            example_similarity_search()
        else:
            print("\n" + "=" * 60)
            print("Skipping similarity search example (numpy not installed)")
            print("Install with: pip install numpy")
            print("=" * 60)

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
