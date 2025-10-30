# Heroku Inference API Embeddings Support

## Overview

Version 0.5.0 adds full support for the Heroku Inference API `/v1/embeddings` endpoint, allowing you to generate vector embeddings for text using Heroku's managed embedding models.

See official documentation: https://devcenter.heroku.com/articles/heroku-inference-api-v1-embeddings

## What's New

### 1. Embeddings Client Type

You can now initialize the LLM client with `type='embedding'` to work with embeddings:

```python
from floship_llm import LLM

llm = LLM(
    type='embedding',
    model='cohere-embed-multilingual'
)

# Generate single embedding
embedding = llm.embed("Hello, world!")  # Returns: List[float]

# Generate multiple embeddings (batch)
embeddings = llm.embed(["Text 1", "Text 2", "Text 3"])  # Returns: List[List[float]]
```

### 2. Input Types for Optimization

Optimize embeddings for specific use cases:

- **`search_document`** - For documents you want to search through
- **`search_query`** - For search queries against document embeddings
- **`classification`** - For training classifiers
- **`clustering`** - For grouping similar items

```python
# For indexing documents
doc_llm = LLM(type='embedding', input_type='search_document')
doc_embeddings = doc_llm.embed(["Document 1", "Document 2"])

# For searching with queries
query_llm = LLM(type='embedding', input_type='search_query')
query_embedding = query_llm.embed("What is...?")
```

### 3. Encoding and Embedding Types

Control output format and precision:

```python
# Float embeddings (default)
llm = LLM(
    type='embedding',
    encoding_format='float',  # or 'base64'
    embedding_type='float'    # float, int8, uint8, binary, ubinary
)
```

### 4. Full Response with Metadata

Get token usage and other metadata:

```python
response = llm.embed("Sample text", return_full_response=True)

print(response['model'])         # Model name
print(response['usage'])         # Token usage info
print(response['data'])          # Embedding objects
```

### 5. Input Validation

- Maximum 96 strings per request
- Maximum 2048 characters per string
- Recommended: < 512 tokens per string
- Warnings for strings exceeding limits

## New Files

### 1. `floship_llm/schemas.py` - New Schemas

Added three new Pydantic models:
- `EmbeddingData` - Single embedding object
- `EmbeddingUsage` - Token usage information
- `EmbeddingResponse` - Full API response

### 2. `example_embeddings.py` - Comprehensive Examples

8 complete examples demonstrating:
1. Basic single text embedding
2. Multiple embeddings (batch)
3. Full response with metadata
4. Search document embeddings
5. Search query embeddings
6. Classification embeddings
7. Clustering embeddings
8. Practical similarity search demo

### 3. `tests/test_embeddings.py` - Test Suite

30 comprehensive tests covering:
- Initialization and parameters
- Single and batch embeddings
- Input validation
- Different input types
- Encoding formats
- Metrics tracking
- Schema validation
- Retry behavior

## Updated Files

### 1. `floship_llm/client.py`

**Changes:**
- Removed the exception that blocked embeddings
- Added embedding-specific parameters: `input_type`, `encoding_format`, `embedding_type`
- Completely rewrote `embed()` method with full API support
- Updated `get_embedding_params()` with all optional parameters
- Added comprehensive validation and error handling
- Added WAF metrics tracking for embeddings

**Key Methods:**
```python
def embed(
    input: Union[str, List[str]],
    return_full_response: bool = False
) -> Union[List[float], List[List[float]], Dict[str, Any]]
```

### 2. `floship_llm/__init__.py`

**Changes:**
- Exported new embedding schemas: `EmbeddingData`, `EmbeddingResponse`, `EmbeddingUsage`
- Updated version to `0.5.0`

### 3. `README.md`

**Changes:**
- Added complete "Embeddings Support" section after streaming documentation
- Documented all parameters, input types, and use cases
- Added practical examples including similarity search
- Included API reference and limits

### 4. `tests/test_client.py`

**Changes:**
- Updated `test_init_embedding_type_not_supported` to `test_init_embedding_type_supported`
- Fixed `test_embed_empty_text` to match new error message

### 5. `tests/test_init.py`

**Changes:**
- Updated version checks from `0.1.1` to `0.5.0`

## API Reference

### Constructor Parameters

```python
LLM(
    type='embedding',                    # Required for embeddings
    model='cohere-embed-multilingual',   # Embedding model ID

    # Optional parameters
    input_type=None,                     # 'search_document', 'search_query',
                                         # 'classification', 'clustering'
    encoding_format='float',             # 'float' or 'base64'
    embedding_type='float',              # 'float', 'int8', 'uint8',
                                         # 'binary', 'ubinary'
    allow_ignored_params=False           # Ignore unsupported parameters
)
```

### embed() Method

```python
embed(
    input: Union[str, List[str]],       # Single text or list (max 96)
    return_full_response: bool = False  # Return full API response
) -> Union[List[float], List[List[float]], Dict]
```

**Returns:**
- Single string input + `return_full_response=False`: `List[float]`
- List input + `return_full_response=False`: `List[List[float]]`
- `return_full_response=True`: `Dict` with full API response

## Testing

All tests pass successfully:
```bash
# Run embedding tests only
pytest tests/test_embeddings.py -v

# Run all tests
pytest tests/ --no-cov
```

**Test Coverage:**
- 30 new embedding-specific tests
- 328 total tests passing
- All existing functionality preserved

## Examples

See `example_embeddings.py` for complete working examples including:
- Basic embeddings
- Batch processing
- Similarity search
- All input types
- Full response handling

Run examples:
```bash
# Set environment variables first
export INFERENCE_URL="https://us.inference.heroku.com"
export INFERENCE_MODEL_ID="cohere-embed-multilingual"
export INFERENCE_KEY="your-api-key"

# Run examples
python example_embeddings.py
```

## Backward Compatibility

✅ All existing functionality is preserved
✅ All existing tests pass without modification (except version updates)
✅ No breaking changes to the API

## Features Inherited

Embeddings inherit these existing features:
- ✅ **Retry mechanism** with exponential backoff
- ✅ **CloudFront WAF protection** (if enabled)
- ✅ **Metrics tracking** (total requests, failures, etc.)
- ✅ **Environment variable configuration**

## Migration Notes

If you previously saw errors about embeddings not being supported:

**Before:**
```python
llm = LLM(type='embedding')  # ❌ Exception: Embedding model is not supported yet
```

**After:**
```python
llm = LLM(type='embedding')  # ✅ Works!
embedding = llm.embed("Hello world")
```

## Version Bump

**0.4.0 → 0.5.0**

This is a minor version bump because:
- New feature added (embeddings support)
- No breaking changes
- Backward compatible
- Follows semantic versioning

## Documentation Links

- [Heroku Embeddings API Docs](https://devcenter.heroku.com/articles/heroku-inference-api-v1-embeddings)
- [README - Embeddings Section](README.md#embeddings-support)
- [Example Code](example_embeddings.py)
- [Test Suite](tests/test_embeddings.py)

## Future Enhancements

Potential future improvements:
- Built-in vector similarity functions
- Integration with vector databases
- Caching for frequently embedded texts
- Batch size auto-optimization
- Progress tracking for large batches
