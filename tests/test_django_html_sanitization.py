"""Test Django HTML sanitization patterns."""

from floship_llm.client import CloudFrontWAFSanitizer


class TestDjangoHTMLSanitization:
    def test_format_html_sanitization(self):
        """Test that format_html() calls are sanitized."""
        content = """
        return format_html('<a href="{}">{}</a>', url, obj.vas_upload)
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized
        assert "format_html_func(" in sanitized
        assert "format_html(" not in sanitized

    def test_html_entities_sanitization(self):
        """Test that HTML entities are sanitized."""
        content = "Error linking VASImportRecord: &quot;test&quot; &lt;span&gt;"
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized
        assert "[QUOT]" in sanitized
        assert "[LT]" in sanitized
        assert "&quot;" not in sanitized
        assert "&lt;" not in sanitized

    def test_html_tags_in_code_sanitization(self):
        """Test that HTML tags in code are sanitized."""
        content = """
        return format_html('<span style="color:red">Error: {}</span>', str(e))
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized
        assert "[SPAN_STYLE]" in sanitized
        assert "[/SPAN]" in sanitized
        assert "<span style=" not in sanitized
        assert "</span>" not in sanitized

    def test_anchor_tags_sanitization(self):
        """Test that anchor tags are sanitized."""
        content = """
        return format_html('<a href="{}">Link</a>', url)
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized
        assert "[A_HREF]" in sanitized
        assert "[/A]" in sanitized
        assert "<a href=" not in sanitized
        assert "</a>" not in sanitized

    def test_heading_tags_sanitization(self):
        """Test that heading tags are sanitized."""
        content = "<H1>Title</H1><H2>Subtitle</H2>"
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized
        assert "[HEADING]" in sanitized
        assert "[/HEADING]" in sanitized
        assert "<H1>" not in sanitized
        assert "</H1>" not in sanitized

    def test_pr_description_with_django_code(self):
        """Test realistic PR description with Django admin code."""
        content = """
        def get_vas_upload_link(self, obj):
            if obj.vas_upload:
                try:
                    url = reverse("admin:accounting_vasimportrecord_change", args=[obj.vas_upload.pk])
                    return format_html('<a href="{}">{}</a>', url, obj.vas_upload)
                except Exception as e:
                    return format_html(
                        '<span style="color:red">Error linking VASImportRecord: {}</span>',
                        str(e),
                    )
            return "-"
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized
        assert "format_html_func(" in sanitized
        assert "[A_HREF]" in sanitized
        assert "[/A]" in sanitized
        assert "[SPAN_STYLE]" in sanitized
        assert "[/SPAN]" in sanitized

    def test_desanitize_restores_django_html(self):
        """Test that desanitization restores Django HTML patterns."""
        original = """
        return format_html('<a href="{}">Link</a>', url)
        """
        sanitized, _ = CloudFrontWAFSanitizer.sanitize(original)
        desanitized, was_desanitized = CloudFrontWAFSanitizer.desanitize(sanitized)

        assert was_desanitized
        assert "format_html(" in desanitized
        assert "<a href=" in desanitized
        assert "</a>" in desanitized

    def test_roundtrip_django_code(self):
        """Test full roundtrip of Django code sanitization/desanitization."""
        original = (
            "return format_html('<span style=\"color:red\">&quot;Error&quot;</span>')"
        )
        sanitized, _ = CloudFrontWAFSanitizer.sanitize(original)
        desanitized, _ = CloudFrontWAFSanitizer.desanitize(sanitized)

        # Verify patterns were sanitized
        assert "format_html_func(" in sanitized
        assert "[SPAN_STYLE]" in sanitized
        assert "[QUOT]" in sanitized

        # Verify restoration
        assert "format_html(" in desanitized
        assert "<span style=" in desanitized
        # v0.5.40: HTML entities now desanitize to actual characters, not HTML entities
        # So [QUOT] becomes " not &quot;
        assert '"Error"' in desanitized
