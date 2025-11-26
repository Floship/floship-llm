"""Tests for the __init__ module."""

import pytest

import floship_llm
from floship_llm import (
    LLM,
    JSONUtils,
    Labels,
    Suggestion,
    SuggestionsResponse,
    ThinkingModel,
    __all__,
    __version__,
    lm_json_utils,
)
from floship_llm.client import LLM as ClientLLM
from floship_llm.schemas import Labels as SchemasLabels
from floship_llm.schemas import Suggestion as SchemasSuggestion
from floship_llm.schemas import SuggestionsResponse as SchemasSuggestionsResponse
from floship_llm.schemas import ThinkingModel as SchemasThinkingModel
from floship_llm.utils import JSONUtils as UtilsJSONUtils
from floship_llm.utils import lm_json_utils as utils_lm_json_utils


class TestModuleImports:
    """Test cases for module imports and exports."""

    def test_version_is_string(self):
        """Test that __version__ is a string."""
        assert isinstance(__version__, str)
        assert __version__ == "0.5.9"

    def test_version_accessible_from_module(self):
        """Test that version is accessible from the main module."""
        assert hasattr(floship_llm, "__version__")
        assert floship_llm.__version__ == "0.5.9"

    def test_all_exports_defined(self):
        """Test that __all__ is properly defined."""
        assert isinstance(__all__, list)

        expected_exports = [
            "LLM",
            "LLMConfig",
            "LLMMetrics",
            "CloudFrontWAFSanitizer",
            "TruncatedResponseError",
            "ContentProcessor",
            "RetryHandler",
            "ToolCall",
            "ToolFunction",
            "ToolParameter",
            "ToolResult",
            "ToolManager",
            "JSONUtils",
            "lm_json_utils",
            "Labels",
            "Suggestion",
            "SuggestionsResponse",
            "ThinkingModel",
            "EmbeddingData",
            "EmbeddingResponse",
            "EmbeddingUsage",
        ]

        assert set(__all__) == set(expected_exports)

    def test_all_exports_are_importable(self):
        """Test that all items in __all__ can be imported."""
        for item in __all__:
            assert hasattr(floship_llm, item), f"Item '{item}' not found in module"

    def test_llm_import(self):
        """Test LLM class import."""
        assert LLM is ClientLLM
        assert hasattr(floship_llm, "LLM")
        assert floship_llm.LLM is ClientLLM

    def test_thinking_model_import(self):
        """Test ThinkingModel import."""
        assert ThinkingModel is SchemasThinkingModel
        assert hasattr(floship_llm, "ThinkingModel")
        assert floship_llm.ThinkingModel is SchemasThinkingModel

    def test_suggestion_import(self):
        """Test Suggestion import."""
        assert Suggestion is SchemasSuggestion
        assert hasattr(floship_llm, "Suggestion")
        assert floship_llm.Suggestion is SchemasSuggestion

    def test_suggestions_response_import(self):
        """Test SuggestionsResponse import."""
        assert SuggestionsResponse is SchemasSuggestionsResponse
        assert hasattr(floship_llm, "SuggestionsResponse")
        assert floship_llm.SuggestionsResponse is SchemasSuggestionsResponse

    def test_labels_import(self):
        """Test Labels import."""
        assert Labels is SchemasLabels
        assert hasattr(floship_llm, "Labels")
        assert floship_llm.Labels is SchemasLabels

    def test_lm_json_utils_import(self):
        """Test lm_json_utils import."""
        assert lm_json_utils is utils_lm_json_utils
        assert hasattr(floship_llm, "lm_json_utils")
        assert floship_llm.lm_json_utils is utils_lm_json_utils

    def test_json_utils_import(self):
        """Test JSONUtils class import."""
        assert JSONUtils is UtilsJSONUtils
        assert hasattr(floship_llm, "JSONUtils")
        assert floship_llm.JSONUtils is UtilsJSONUtils


class TestModuleDocstring:
    """Test cases for module docstring and metadata."""

    def test_module_has_docstring(self):
        """Test that module has a docstring."""
        assert floship_llm.__doc__ is not None
        assert len(floship_llm.__doc__.strip()) > 0

    def test_docstring_content(self):
        """Test docstring content."""
        docstring = floship_llm.__doc__
        assert "Floship LLM Client Library" in docstring
        assert "reusable LLM client library" in docstring
        assert "OpenAI-compatible inference endpoints" in docstring


class TestImportedClassesFunctionality:
    """Test that imported classes work correctly."""

    def test_thinking_model_instantiation(self):
        """Test that ThinkingModel can be instantiated."""
        model = ThinkingModel(thinking="Test thinking")
        assert model.thinking == "Test thinking"

    def test_suggestion_instantiation(self):
        """Test that Suggestion can be instantiated."""
        suggestion = Suggestion(
            thinking="Test thinking",
            file_path="/test/path.py",
            line=1,
            suggestion="```suggestion\\ntest\\n```",
            severity=5,
            type="bug",
            reason="Test reason",
        )
        assert suggestion.thinking == "Test thinking"
        assert suggestion.file_path == "/test/path.py"
        assert suggestion.type == "bug"

    def test_suggestions_response_instantiation(self):
        """Test that SuggestionsResponse can be instantiated."""
        suggestion = Suggestion(
            thinking="Individual thinking",
            file_path="/test/path.py",
            line=1,
            suggestion="```suggestion\\ntest\\n```",
            severity=5,
            type="bug",
            reason="Test reason",
        )

        response = SuggestionsResponse(
            thinking="Overall thinking", suggestions=[suggestion]
        )

        assert response.thinking == "Overall thinking"
        assert len(response.suggestions) == 1
        assert response.suggestions[0].type == "bug"

    def test_labels_instantiation(self):
        """Test that Labels can be instantiated."""
        labels = Labels(thinking="Test thinking", labels=["bug", "priority-high"])
        assert labels.thinking == "Test thinking"
        assert labels.labels == ["bug", "priority-high"]

    def test_json_utils_functionality(self):
        """Test that JSONUtils works correctly."""
        utils = JSONUtils()

        text = 'Text {"key": "value"} end'
        result = utils.extract_and_fix_json(text)

        assert len(result) == 1
        assert result[0] == {"key": "value"}

    def test_lm_json_utils_functionality(self):
        """Test that lm_json_utils instance works correctly."""
        text = 'Text {"key": "value"} end'
        result = lm_json_utils.extract_and_fix_json(text)

        assert len(result) == 1
        assert result[0] == {"key": "value"}


class TestWildcardImports:
    """Test wildcard imports work correctly."""

    def test_wildcard_import_includes_all_exports(self):
        """Test that wildcard import includes all __all__ exports."""
        # This test verifies that when someone does 'from floship_llm import *'
        # they get all the items listed in __all__

        import floship_llm

        # Get all public attributes (those in __all__)
        public_attrs = {
            name: getattr(floship_llm, name) for name in floship_llm.__all__
        }

        # Verify all expected items are present
        expected_items = [
            "LLM",
            "ThinkingModel",
            "Suggestion",
            "SuggestionsResponse",
            "Labels",
            "lm_json_utils",
            "JSONUtils",
        ]

        for item in expected_items:
            assert item in public_attrs, f"'{item}' missing from public attributes"
            assert public_attrs[item] is not None, f"'{item}' is None"

    def test_no_private_exports_in_all(self):
        """Test that __all__ doesn't contain private attributes."""
        for item in __all__:
            assert not item.startswith(
                "_"
            ), f"Private attribute '{item}' found in __all__"

    def test_all_matches_actual_exports(self):
        """Test that __all__ matches what's actually exported."""
        # Get all non-private attributes from the module
        actual_exports = [
            name
            for name in dir(floship_llm)
            if not name.startswith("_") and name not in ["__version__"]
        ]  # Exclude special cases

        # __all__ should be a subset of actual exports
        for item in __all__:
            assert (
                item in actual_exports
            ), f"'{item}' in __all__ but not actually exported"


class TestModuleStructure:
    """Test module structure and organization."""

    def test_module_has_proper_attributes(self):
        """Test that module has expected attributes."""
        expected_attrs = ["__doc__", "__version__", "__all__"]

        for attr in expected_attrs:
            assert hasattr(floship_llm, attr), f"Module missing '{attr}' attribute"

    def test_imports_are_not_duplicated(self):
        """Test that imports are not duplicated in module namespace."""
        # Check that we don't have both 'LLM' and something like 'client_LLM'
        module_attrs = dir(floship_llm)

        # Filter to only public attributes
        public_attrs = [attr for attr in module_attrs if not attr.startswith("_")]

        # Should only have the items in __all__ plus any additional public items
        # The key point is we shouldn't have duplicate references to the same class
        llm_refs = [attr for attr in public_attrs if "LLM" in attr]

        # Should only have one LLM reference
        assert "LLM" in llm_refs

        # Might have others like 'JSONUtils' but that's expected


class TestBackwardsCompatibility:
    """Test backwards compatibility aspects."""

    def test_direct_import_paths_work(self):
        """Test that direct import paths still work."""
        # These should work for backwards compatibility
        from floship_llm.client import LLM
        from floship_llm.schemas import (
            Labels,
            Suggestion,
            SuggestionsResponse,
            ThinkingModel,
        )
        from floship_llm.utils import JSONUtils, lm_json_utils

        # Verify they're the same as the re-exported ones
        assert LLM is floship_llm.LLM
        assert ThinkingModel is floship_llm.ThinkingModel
        assert Suggestion is floship_llm.Suggestion
        assert SuggestionsResponse is floship_llm.SuggestionsResponse
        assert Labels is floship_llm.Labels
        assert JSONUtils is floship_llm.JSONUtils
        assert lm_json_utils is floship_llm.lm_json_utils

    def test_module_import_style(self):
        """Test different import styles work."""
        # Style 1: Import module
        import floship_llm

        assert hasattr(floship_llm, "LLM")

        # Style 2: Import specific items
        from floship_llm import LLM, ThinkingModel

        assert LLM is not None
        assert ThinkingModel is not None

        # Style 3: Import with alias
        from floship_llm import LLM as FloshipLLM

        assert FloshipLLM is not None
