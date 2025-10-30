# Release Notes: floship-llm v0.5.0

**Release Date:** October 30, 2025
**Type:** Minor Release (Feature Addition)
**Status:** ✅ Ready for Release

## 🎉 What's New

### Major Feature: Full Heroku Inference API Embeddings Support

Version 0.5.0 adds complete support for the Heroku Inference API `/v1/embeddings` endpoint, enabling vector embeddings generation for search, classification, clustering, and other ML tasks.

## 📦 Package Information

- **Package Name:** `floship-llm`
- **Version:** `0.5.0`
- **Python Support:** 3.8, 3.9, 3.10, 3.11, 3.12
- **License:** MIT
- **Repository:** https://github.com/Floship/floship-llm

## ✨ New Features

### 1. Embeddings Client Type
```python
from floship_llm import LLM

llm = LLM(type='embedding', model='cohere-embed-multilingual')
embedding = llm.embed("Hello, world!")
```

### 2. Batch Processing
```python
embeddings = llm.embed(["Text 1", "Text 2", "Text 3"])
# Returns: List[List[float]]
```

### 3. Input Type Optimization
- `search_document` - For indexing documents
- `search_query` - For search queries
- `classification` - For training classifiers
- `clustering` - For grouping similar items

### 4. Full Response Mode
```python
response = llm.embed("Text", return_full_response=True)
print(response['usage'])  # Token usage info
```

### 5. Encoding Options
- `encoding_format`: 'float' or 'base64'
- `embedding_type`: 'float', 'int8', 'uint8', 'binary', 'ubinary'

## 📚 New Files

1. **`example_embeddings.py`** - 8 comprehensive examples
   - Basic and batch embeddings
   - All input types
   - Similarity search demo

2. **`tests/test_embeddings.py`** - 30 new tests
   - Complete test coverage for embeddings

3. **`EMBEDDINGS_FEATURE.md`** - Detailed feature documentation

## 🔧 Modified Files

- `floship_llm/client.py` - Full embeddings implementation
- `floship_llm/schemas.py` - New embedding schemas
- `floship_llm/__init__.py` - Version bump and exports
- `README.md` - Comprehensive embeddings documentation
- `CHANGELOG.md` - v0.5.0 release notes
- `pyproject.toml` - Version update to 0.5.0
- `tests/test_client.py` - Updated for embeddings support
- `tests/test_init.py` - Version checks updated

## ✅ Quality Assurance

- ✅ **328 tests passing** (30 new embedding tests)
- ✅ **100% backward compatible** - No breaking changes
- ✅ **Comprehensive documentation** - README, examples, docstrings
- ✅ **Working examples** - Tested and verified
- ✅ **Type hints** - Full typing support
- ✅ **Error handling** - Robust validation

## 🚀 Installation

```bash
# From PyPI (after release)
pip install floship-llm==0.5.0

# From source
git clone https://github.com/Floship/floship-llm.git
cd floship-llm
pip install -e .
```

## 📖 Quick Start

```python
from floship_llm import LLM

# Set up environment variables
# export INFERENCE_URL="https://us.inference.heroku.com"
# export INFERENCE_MODEL_ID="cohere-embed-multilingual"
# export INFERENCE_KEY="your-api-key"

# Create embeddings client
llm = LLM(type='embedding')

# Generate embeddings
embedding = llm.embed("Hello, world!")
print(f"Dimension: {len(embedding)}")

# Batch processing
embeddings = llm.embed(["Text 1", "Text 2"])
print(f"Generated {len(embeddings)} embeddings")
```

## 🔗 Documentation

- [README - Embeddings Section](README.md#embeddings-support)
- [Feature Documentation](EMBEDDINGS_FEATURE.md)
- [Example Code](example_embeddings.py)
- [Test Suite](tests/test_embeddings.py)
- [Heroku API Docs](https://devcenter.heroku.com/articles/heroku-inference-api-v1-embeddings)

## 🔄 Migration Guide

### From 0.4.0 to 0.5.0

**No breaking changes!** All existing code continues to work.

**New capability:**
```python
# Previously raised exception
llm = LLM(type='embedding')  # ❌ "Embedding model is not supported yet"

# Now works!
llm = LLM(type='embedding')  # ✅ Full support
embedding = llm.embed("Hello world")
```

## 🎯 Features Inherited

Embeddings automatically benefit from existing features:
- ✅ Retry mechanism with exponential backoff
- ✅ CloudFront WAF protection
- ✅ Metrics tracking
- ✅ Environment variable configuration
- ✅ Comprehensive error handling

## 📊 Test Results

```
Tests: 328 passed in 0.86s
Coverage: Complete for new features
Warnings: 9 (Pydantic deprecations, non-blocking)
```

## 🐛 Known Issues

None - This is a stable release.

## 🔮 Future Enhancements

Potential improvements for future versions:
- Built-in vector similarity functions
- Integration with vector databases
- Caching for frequently embedded texts
- Batch size auto-optimization
- Progress tracking for large batches

## 👥 Contributors

- Floship Development Team

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Heroku for the Inference API
- OpenAI for the Python SDK
- Pydantic for data validation

---

## Release Checklist

- [x] All tests passing (328/328)
- [x] Version updated in `pyproject.toml` (0.5.0)
- [x] Version updated in `__init__.py` (0.5.0)
- [x] CHANGELOG.md updated with release notes
- [x] README.md updated with new features
- [x] Examples created and tested
- [x] Test suite expanded (30 new tests)
- [x] Documentation complete
- [x] Backward compatibility verified
- [x] No breaking changes
- [ ] Git tag created (`git tag v0.5.0`)
- [ ] Push to GitHub (`git push origin main --tags`)
- [ ] Build package (`python -m build`)
- [ ] Upload to PyPI (`python -m twine upload dist/*`)

## Release Commands

```bash
# 1. Commit all changes
git add .
git commit -m "Release v0.5.0: Add full Heroku Inference API embeddings support"

# 2. Create and push tag
git tag -a v0.5.0 -m "Version 0.5.0: Full embeddings support"
git push origin main --tags

# 3. Build package
python -m build

# 4. Upload to PyPI (production)
python -m twine upload dist/*

# Or upload to Test PyPI first
python -m twine upload --repository testpypi dist/*
```

---

**Ready for Release! 🚀**
