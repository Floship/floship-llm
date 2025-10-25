"""Tests for the schemas module."""

import pytest
from pydantic import ValidationError

from floship_llm.schemas import Labels, Suggestion, SuggestionsResponse, ThinkingModel


class TestThinkingModel:
    """Test cases for the ThinkingModel schema."""

    def test_thinking_model_valid(self):
        """Test ThinkingModel with valid data."""
        data = {"thinking": "This is my thought process"}
        model = ThinkingModel(**data)

        assert model.thinking == "This is my thought process"

    def test_thinking_model_missing_thinking(self):
        """Test ThinkingModel with missing thinking field."""
        with pytest.raises(ValidationError) as exc_info:
            ThinkingModel()

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("thinking",)

    def test_thinking_model_empty_thinking(self):
        """Test ThinkingModel with empty thinking field."""
        data = {"thinking": ""}
        model = ThinkingModel(**data)

        assert model.thinking == ""

    def test_thinking_model_extra_fields(self):
        """Test ThinkingModel with extra fields."""
        data = {"thinking": "My thoughts", "extra_field": "should be ignored"}
        model = ThinkingModel(**data)

        assert model.thinking == "My thoughts"
        assert not hasattr(model, "extra_field")

    def test_thinking_model_json_serialization(self):
        """Test ThinkingModel JSON serialization."""
        data = {"thinking": "Test thinking"}
        model = ThinkingModel(**data)

        json_data = model.model_dump()
        assert json_data == {"thinking": "Test thinking"}

    def test_thinking_model_from_json(self):
        """Test ThinkingModel creation from JSON."""
        json_str = '{"thinking": "From JSON"}'
        model = ThinkingModel.model_validate_json(json_str)

        assert model.thinking == "From JSON"


class TestSuggestion:
    """Test cases for the Suggestion schema."""

    def test_suggestion_valid(self):
        """Test Suggestion with valid data."""
        data = {
            "thinking": "Need to fix this bug",
            "file_path": "/path/to/file.py",
            "line": 42,
            "suggestion": "```suggestion\\nfixed code\\n```",
            "severity": 5,
            "type": "bug",
            "reason": "This will fix the issue",
        }
        suggestion = Suggestion(**data)

        assert suggestion.thinking == "Need to fix this bug"
        assert suggestion.file_path == "/path/to/file.py"
        assert suggestion.line == 42
        assert suggestion.suggestion == "```suggestion\\nfixed code\\n```"
        assert suggestion.severity == 5
        assert suggestion.type == "bug"
        assert suggestion.reason == "This will fix the issue"

    def test_suggestion_inherits_thinking(self):
        """Test that Suggestion inherits from ThinkingModel."""
        assert issubclass(Suggestion, ThinkingModel)

    def test_suggestion_missing_required_fields(self):
        """Test Suggestion with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            Suggestion()

        errors = exc_info.value.errors()
        required_fields = {
            "thinking",
            "file_path",
            "line",
            "suggestion",
            "severity",
            "type",
            "reason",
        }
        error_fields = {error["loc"][0] for error in errors}

        assert required_fields.issubset(error_fields)

    def test_suggestion_invalid_line_type(self):
        """Test Suggestion with invalid line type."""
        data = {
            "thinking": "Test",
            "file_path": "/path/to/file.py",
            "line": "not_a_number",  # Should be int
            "suggestion": "```suggestion\\ncode\\n```",
            "severity": 5,
            "type": "bug",
            "reason": "Fix",
        }

        with pytest.raises(ValidationError) as exc_info:
            Suggestion(**data)

        errors = exc_info.value.errors()
        line_errors = [e for e in errors if e["loc"][0] == "line"]
        assert len(line_errors) > 0
        assert line_errors[0]["type"] == "int_parsing"

    def test_suggestion_invalid_severity_type(self):
        """Test Suggestion with invalid severity type."""
        data = {
            "thinking": "Test",
            "file_path": "/path/to/file.py",
            "line": 1,
            "suggestion": "```suggestion\\ncode\\n```",
            "severity": "high",  # Should be int
            "type": "bug",
            "reason": "Fix",
        }

        with pytest.raises(ValidationError) as exc_info:
            Suggestion(**data)

        errors = exc_info.value.errors()
        severity_errors = [e for e in errors if e["loc"][0] == "severity"]
        assert len(severity_errors) > 0
        assert severity_errors[0]["type"] == "int_parsing"

    @pytest.mark.parametrize(
        "suggestion_type",
        ["bug", "neatpick", "text_change", "refactor", "performance", "security"],
    )
    def test_suggestion_valid_types(self, suggestion_type):
        """Test Suggestion with various valid types."""
        data = {
            "thinking": "Test thinking",
            "file_path": "/path/to/file.py",
            "line": 1,
            "suggestion": "```suggestion\\ncode\\n```",
            "severity": 5,
            "type": suggestion_type,
            "reason": "Test reason",
        }
        suggestion = Suggestion(**data)
        assert suggestion.type == suggestion_type

    def test_suggestion_json_serialization(self):
        """Test Suggestion JSON serialization."""
        data = {
            "thinking": "Test thinking",
            "file_path": "/path/to/file.py",
            "line": 1,
            "suggestion": "```suggestion\\ncode\\n```",
            "severity": 5,
            "type": "bug",
            "reason": "Test reason",
        }
        suggestion = Suggestion(**data)

        json_data = suggestion.model_dump()
        assert json_data == data

    def test_suggestion_from_json(self):
        """Test Suggestion creation from JSON."""
        json_str = """
        {
            "thinking": "Test thinking",
            "file_path": "/path/to/file.py",
            "line": 1,
            "suggestion": "```suggestion\\ncode\\n```",
            "severity": 5,
            "type": "bug",
            "reason": "Test reason"
        }
        """
        suggestion = Suggestion.model_validate_json(json_str)

        assert suggestion.thinking == "Test thinking"
        assert suggestion.file_path == "/path/to/file.py"
        assert suggestion.line == 1


class TestSuggestionsResponse:
    """Test cases for the SuggestionsResponse schema."""

    def test_suggestions_response_valid(self):
        """Test SuggestionsResponse with valid data."""
        suggestion_data = {
            "thinking": "Individual suggestion thinking",
            "file_path": "/path/to/file.py",
            "line": 1,
            "suggestion": "```suggestion\\ncode\\n```",
            "severity": 5,
            "type": "bug",
            "reason": "Test reason",
        }

        data = {
            "thinking": "Overall thinking about suggestions",
            "suggestions": [suggestion_data],
        }

        response = SuggestionsResponse(**data)

        assert response.thinking == "Overall thinking about suggestions"
        assert len(response.suggestions) == 1
        assert isinstance(response.suggestions[0], Suggestion)
        assert response.suggestions[0].file_path == "/path/to/file.py"

    def test_suggestions_response_inherits_thinking(self):
        """Test that SuggestionsResponse inherits from ThinkingModel."""
        assert issubclass(SuggestionsResponse, ThinkingModel)

    def test_suggestions_response_empty_suggestions(self):
        """Test SuggestionsResponse with empty suggestions list."""
        data = {"thinking": "No suggestions found", "suggestions": []}

        response = SuggestionsResponse(**data)

        assert response.thinking == "No suggestions found"
        assert len(response.suggestions) == 0

    def test_suggestions_response_multiple_suggestions(self):
        """Test SuggestionsResponse with multiple suggestions."""
        suggestion1 = {
            "thinking": "First suggestion thinking",
            "file_path": "/path/to/file1.py",
            "line": 1,
            "suggestion": "```suggestion\\ncode1\\n```",
            "severity": 5,
            "type": "bug",
            "reason": "First reason",
        }

        suggestion2 = {
            "thinking": "Second suggestion thinking",
            "file_path": "/path/to/file2.py",
            "line": 10,
            "suggestion": "```suggestion\\ncode2\\n```",
            "severity": 3,
            "type": "refactor",
            "reason": "Second reason",
        }

        data = {
            "thinking": "Multiple suggestions found",
            "suggestions": [suggestion1, suggestion2],
        }

        response = SuggestionsResponse(**data)

        assert response.thinking == "Multiple suggestions found"
        assert len(response.suggestions) == 2
        assert all(isinstance(s, Suggestion) for s in response.suggestions)
        assert response.suggestions[0].type == "bug"
        assert response.suggestions[1].type == "refactor"

    def test_suggestions_response_missing_thinking(self):
        """Test SuggestionsResponse with missing thinking field."""
        data = {"suggestions": []}

        with pytest.raises(ValidationError) as exc_info:
            SuggestionsResponse(**data)

        errors = exc_info.value.errors()
        thinking_errors = [e for e in errors if e["loc"][0] == "thinking"]
        assert len(thinking_errors) > 0
        assert thinking_errors[0]["type"] == "missing"

    def test_suggestions_response_missing_suggestions(self):
        """Test SuggestionsResponse with missing suggestions field."""
        data = {"thinking": "Test thinking"}

        with pytest.raises(ValidationError) as exc_info:
            SuggestionsResponse(**data)

        errors = exc_info.value.errors()
        suggestions_errors = [e for e in errors if e["loc"][0] == "suggestions"]
        assert len(suggestions_errors) > 0
        assert suggestions_errors[0]["type"] == "missing"

    def test_suggestions_response_invalid_suggestion_item(self):
        """Test SuggestionsResponse with invalid suggestion item."""
        data = {
            "thinking": "Test thinking",
            "suggestions": [
                {
                    "thinking": "Valid suggestion",
                    "file_path": "/path/to/file.py",
                    "line": 1,
                    "suggestion": "```suggestion\\ncode\\n```",
                    "severity": 5,
                    "type": "bug",
                    "reason": "Valid reason",
                },
                {
                    # Missing required fields
                    "thinking": "Invalid suggestion"
                },
            ],
        }

        with pytest.raises(ValidationError) as exc_info:
            SuggestionsResponse(**data)

        errors = exc_info.value.errors()
        # Should have errors for the second suggestion
        suggestion_errors = [
            e for e in errors if e["loc"][0] == "suggestions" and e["loc"][1] == 1
        ]
        assert len(suggestion_errors) > 0

    def test_suggestions_response_json_serialization(self):
        """Test SuggestionsResponse JSON serialization."""
        suggestion_data = {
            "thinking": "Individual suggestion thinking",
            "file_path": "/path/to/file.py",
            "line": 1,
            "suggestion": "```suggestion\\ncode\\n```",
            "severity": 5,
            "type": "bug",
            "reason": "Test reason",
        }

        data = {"thinking": "Overall thinking", "suggestions": [suggestion_data]}

        response = SuggestionsResponse(**data)
        json_data = response.model_dump()

        assert json_data["thinking"] == "Overall thinking"
        assert len(json_data["suggestions"]) == 1
        assert json_data["suggestions"][0] == suggestion_data


class TestLabels:
    """Test cases for the Labels schema."""

    def test_labels_valid(self):
        """Test Labels with valid data."""
        data = {
            "thinking": "These labels apply to the ticket",
            "labels": ["bug", "priority-high", "backend"],
        }

        labels = Labels(**data)

        assert labels.thinking == "These labels apply to the ticket"
        assert labels.labels == ["bug", "priority-high", "backend"]

    def test_labels_inherits_thinking(self):
        """Test that Labels inherits from ThinkingModel."""
        assert issubclass(Labels, ThinkingModel)

    def test_labels_empty_list(self):
        """Test Labels with empty labels list."""
        data = {"thinking": "No labels needed", "labels": []}

        labels = Labels(**data)

        assert labels.thinking == "No labels needed"
        assert labels.labels == []

    def test_labels_single_label(self):
        """Test Labels with single label."""
        data = {"thinking": "Only one label needed", "labels": ["bug"]}

        labels = Labels(**data)

        assert labels.thinking == "Only one label needed"
        assert labels.labels == ["bug"]

    def test_labels_max_five_labels(self):
        """Test Labels with exactly five labels."""
        data = {
            "thinking": "Maximum allowed labels",
            "labels": ["bug", "high-priority", "backend", "urgent", "security"],
        }

        labels = Labels(**data)

        assert labels.thinking == "Maximum allowed labels"
        assert len(labels.labels) == 5
        assert labels.labels == [
            "bug",
            "high-priority",
            "backend",
            "urgent",
            "security",
        ]

    def test_labels_duplicate_labels(self):
        """Test Labels with duplicate labels (should be allowed by schema but noted in description)."""
        data = {
            "thinking": "Some duplicate labels",
            "labels": ["bug", "bug", "priority-high"],
        }

        # Pydantic doesn't prevent duplicates by default, but the description mentions "unique labels"
        labels = Labels(**data)
        assert labels.labels == ["bug", "bug", "priority-high"]

    def test_labels_missing_thinking(self):
        """Test Labels with missing thinking field."""
        data = {"labels": ["bug"]}

        with pytest.raises(ValidationError) as exc_info:
            Labels(**data)

        errors = exc_info.value.errors()
        thinking_errors = [e for e in errors if e["loc"][0] == "thinking"]
        assert len(thinking_errors) > 0
        assert thinking_errors[0]["type"] == "missing"

    def test_labels_missing_labels(self):
        """Test Labels with missing labels field."""
        data = {"thinking": "Test thinking"}

        with pytest.raises(ValidationError) as exc_info:
            Labels(**data)

        errors = exc_info.value.errors()
        labels_errors = [e for e in errors if e["loc"][0] == "labels"]
        assert len(labels_errors) > 0
        assert labels_errors[0]["type"] == "missing"

    def test_labels_invalid_labels_type(self):
        """Test Labels with invalid labels type."""
        data = {"thinking": "Test thinking", "labels": "not-a-list"}  # Should be list

        with pytest.raises(ValidationError) as exc_info:
            Labels(**data)

        errors = exc_info.value.errors()
        labels_errors = [e for e in errors if e["loc"][0] == "labels"]
        assert len(labels_errors) > 0
        assert labels_errors[0]["type"] == "list_type"

    def test_labels_non_string_label_items(self):
        """Test Labels with non-string items in labels list."""
        data = {
            "thinking": "Test thinking",
            "labels": [
                "bug",
                123,
                "priority-high",
            ],  # 123 should cause validation error
        }

        with pytest.raises(ValidationError) as exc_info:
            Labels(**data)

        errors = exc_info.value.errors()
        label_errors = [e for e in errors if "labels" in str(e["loc"])]
        assert len(label_errors) > 0

    def test_labels_json_serialization(self):
        """Test Labels JSON serialization."""
        data = {"thinking": "Test thinking", "labels": ["bug", "priority-high"]}

        labels = Labels(**data)
        json_data = labels.model_dump()

        assert json_data == data

    def test_labels_from_json(self):
        """Test Labels creation from JSON."""
        json_str = """
        {
            "thinking": "From JSON",
            "labels": ["bug", "json-test"]
        }
        """

        labels = Labels.model_validate_json(json_str)

        assert labels.thinking == "From JSON"
        assert labels.labels == ["bug", "json-test"]
