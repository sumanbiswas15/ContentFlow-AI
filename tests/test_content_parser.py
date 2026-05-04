"""
Unit tests for the content parsing and validation system.

Tests cover format-specific parsing methods, content validation logic,
error handling, and round-trip consistency.
"""

import json
import xml.etree.ElementTree as ET
import pytest
from unittest.mock import patch

from app.utils.content_parser import (
    ContentParser, ParsedContent, parse_markdown_content,
    parse_html_content, parse_json_content, validate_content_round_trip
)
from app.models.base import ContentFormat, ContentType, ValidationResult
from app.core.exceptions import ContentParsingError, ValidationError


class TestContentParser:
    """Test cases for the ContentParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ContentParser()
    
    def test_parse_markdown_content(self):
        """Test parsing markdown content."""
        markdown_content = """# Test Title

This is a **bold** text with a [link](https://example.com).

```python
def hello():
    return "world"
```

- Item 1
- Item 2
"""
        
        parsed = self.parser.parse_content(
            markdown_content, 
            ContentFormat.MARKDOWN, 
            ContentType.TEXT
        )
        
        assert parsed.is_valid()
        assert parsed.content == markdown_content
        assert parsed.content_format == ContentFormat.MARKDOWN
        assert parsed.content_type == ContentType.TEXT
        assert parsed.metadata['has_code_blocks'] is True
        assert parsed.metadata['has_links'] is True
        assert parsed.metadata['word_count'] > 0
    
    def test_parse_html_content(self):
        """Test parsing HTML content."""
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="Test description">
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a <a href="https://example.com">link</a>.</p>
    <img src="test.jpg" alt="Test image">
</body>
</html>"""
        
        parsed = self.parser.parse_content(
            html_content,
            ContentFormat.HTML,
            ContentType.TEXT
        )
        
        assert parsed.is_valid()
        assert parsed.content == html_content
        assert parsed.metadata['title'] == 'Test Page'
        assert parsed.metadata['has_images'] is True
        assert parsed.metadata['has_links'] is True
        assert 'description' in parsed.metadata['meta_tags']
    
    def test_parse_plain_text_content(self):
        """Test parsing plain text content."""
        text_content = """This is a simple text document.

It has multiple paragraphs and some URLs like https://example.com.
Also an email: test@example.com

The end."""
        
        parsed = self.parser.parse_content(
            text_content,
            ContentFormat.PLAIN_TEXT,
            ContentType.TEXT
        )
        
        assert parsed.is_valid()
        assert parsed.content == text_content
        assert parsed.metadata['has_urls'] is True
        assert parsed.metadata['has_emails'] is True
        assert parsed.metadata['line_count'] == 6
        assert parsed.metadata['word_count'] > 0
    
    def test_parse_json_content(self):
        """Test parsing JSON content."""
        json_data = {
            "name": "Test",
            "value": 42,
            "nested": {
                "array": [1, 2, 3],
                "boolean": True
            }
        }
        json_content = json.dumps(json_data, indent=2)
        
        parsed = self.parser.parse_content(
            json_content,
            ContentFormat.JSON,
            ContentType.TEXT
        )
        
        assert parsed.is_valid()
        assert parsed.content == json_data
        assert parsed.metadata['json_type'] == 'dict'
        assert parsed.metadata['key_count'] == 3
        assert parsed.metadata['has_nested_objects'] is True
        assert parsed.metadata['max_depth'] == 3
    
    def test_parse_xml_content(self):
        """Test parsing XML content."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<root xmlns="http://example.com/ns">
    <item id="1">
        <name>Test Item</name>
        <value>42</value>
        <nested>
            <child>Content</child>
        </nested>
    </item>
</root>"""
        
        parsed = self.parser.parse_content(
            xml_content,
            ContentFormat.XML,
            ContentType.TEXT
        )
        
        assert parsed.is_valid()
        assert isinstance(parsed.content, ET.Element)
        assert parsed.metadata['root_tag'] == '{http://example.com/ns}root'
        assert parsed.metadata['namespace'] == 'http://example.com/ns'
        assert parsed.metadata['element_count'] > 1
        assert parsed.metadata['has_text_content'] is True
    
    def test_parse_empty_content_raises_error(self):
        """Test that empty content raises ContentParsingError."""
        with pytest.raises(ContentParsingError) as exc_info:
            self.parser.parse_content("", ContentFormat.PLAIN_TEXT)
        
        assert "Content cannot be empty" in str(exc_info.value)
    
    def test_parse_unsupported_format_raises_error(self):
        """Test that unsupported format raises ContentParsingError."""
        # Create a mock unsupported format
        with patch.object(ContentFormat, '__contains__', return_value=False):
            with pytest.raises(ContentParsingError) as exc_info:
                self.parser.parse_content("test", "unsupported_format")
            
            assert "Unsupported content format" in str(exc_info.value)
    
    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises ContentParsingError."""
        invalid_json = '{"invalid": json content}'
        
        with pytest.raises(ContentParsingError) as exc_info:
            self.parser.parse_content(invalid_json, ContentFormat.JSON)
        
        assert "Failed to parse json content" in str(exc_info.value)
    
    def test_parse_invalid_xml_raises_error(self):
        """Test that invalid XML raises ContentParsingError."""
        invalid_xml = '<root><unclosed>tag</root>'
        
        with pytest.raises(ContentParsingError) as exc_info:
            self.parser.parse_content(invalid_xml, ContentFormat.XML)
        
        assert "Failed to parse xml content" in str(exc_info.value)
    
    def test_validate_markdown_with_issues(self):
        """Test markdown validation with various issues."""
        problematic_markdown = """# Title

This has an [empty link]() and unmatched code blocks.

```python
def test():
    pass
```

Another code block without closing:
```javascript
console.log("test");

This line is way too long and exceeds the recommended 120 character limit for readability and should trigger a warning about line length issues.
"""
        
        parsed = self.parser.parse_content(
            problematic_markdown,
            ContentFormat.MARKDOWN,
            validate=False  # Parse without validation first
        )
        
        validation_result = self.parser.validate_content(parsed)
        
        assert not validation_result.is_valid
        assert any("Unmatched code block" in error for error in validation_result.errors)
        assert any("Empty URL in link" in warning for warning in validation_result.warnings)
        assert any("longer than 120 characters" in warning for warning in validation_result.warnings)
    
    def test_validate_html_with_issues(self):
        """Test HTML validation with accessibility issues."""
        problematic_html = """<html>
<body>
    <img src="test.jpg">
    <a>Empty link</a>
    <div style="color: red;">Inline style</div>
</body>
</html>"""
        
        parsed = self.parser.parse_content(
            problematic_html,
            ContentFormat.HTML,
            validate=False
        )
        
        validation_result = self.parser.validate_content(parsed)
        
        # Should have warnings but still be valid
        assert validation_result.is_valid
        assert any("without alt attributes" in warning for warning in validation_result.warnings)
        assert any("without href attributes" in warning for warning in validation_result.warnings)
        assert any("inline styles" in warning for warning in validation_result.warnings)
    
    def test_validate_json_with_issues(self):
        """Test JSON validation with depth and size issues."""
        # Create deeply nested JSON
        deep_json = {"level1": {"level2": {"level3": {"level4": {"level5": {}}}}}}
        
        # Mock metadata to simulate large size
        parsed = ParsedContent(
            content=deep_json,
            content_type=ContentType.TEXT,
            content_format=ContentFormat.JSON,
            metadata={'max_depth': 25, 'size_bytes': 2 * 1024 * 1024}  # 2MB
        )
        
        validation_result = self.parser.validate_content(parsed)
        
        assert validation_result.is_valid  # Should still be valid but with warnings
        assert any("very deep" in warning for warning in validation_result.warnings)
        assert any("large" in warning for warning in validation_result.warnings)
    
    def test_format_content_to_different_formats(self):
        """Test formatting content to different output formats."""
        # Start with markdown
        markdown_content = "# Title\n\nThis is **bold** text."
        parsed = self.parser.parse_content(markdown_content, ContentFormat.MARKDOWN)
        
        # Format to HTML
        html_output = self.parser.format_content(parsed, ContentFormat.HTML)
        assert "<h1>Title</h1>" in html_output
        assert "<strong>bold</strong>" in html_output
        
        # Format to plain text
        text_output = self.parser.format_content(parsed, ContentFormat.PLAIN_TEXT)
        assert "Title" in text_output
        assert "bold" in text_output
        
        # Format to JSON
        json_output = self.parser.format_content(parsed, ContentFormat.JSON)
        json_data = json.loads(json_output)
        assert json_data['content'] == markdown_content
        assert json_data['content_format'] == 'markdown'
    
    def test_format_content_unsupported_format_raises_error(self):
        """Test that formatting to unsupported format raises error."""
        parsed = ParsedContent(
            content="test",
            content_type=ContentType.TEXT,
            content_format=ContentFormat.PLAIN_TEXT
        )
        
        with pytest.raises(ContentParsingError) as exc_info:
            self.parser.format_content(parsed, "unsupported_format")
        
        assert "Unsupported target format" in str(exc_info.value)
    
    def test_round_trip_consistency_markdown(self):
        """Test round-trip consistency for markdown content."""
        original_content = """# Test Document

This is a test with **bold** and *italic* text.

- List item 1
- List item 2

[Link](https://example.com)

```python
def test():
    return True
```
"""
        
        # Test round-trip
        parsed1 = self.parser.parse_content(original_content, ContentFormat.MARKDOWN, validate=False)
        formatted = self.parser.format_content(parsed1)
        parsed2 = self.parser.parse_content(formatted, ContentFormat.MARKDOWN, validate=False)
        
        assert parsed1.content.strip() == parsed2.content.strip()
    
    def test_round_trip_consistency_json(self):
        """Test round-trip consistency for JSON content."""
        original_data = {
            "string": "test",
            "number": 42,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "object": {"nested": "value"}
        }
        original_content = json.dumps(original_data, sort_keys=True)
        
        # Test round-trip
        parsed1 = self.parser.parse_content(original_content, ContentFormat.JSON, validate=False)
        formatted = self.parser.format_content(parsed1)
        parsed2 = self.parser.parse_content(formatted, ContentFormat.JSON, validate=False)
        
        assert parsed1.content == parsed2.content
    
    def test_convenience_functions(self):
        """Test convenience functions for common parsing tasks."""
        # Test markdown parsing
        markdown_result = parse_markdown_content("# Test\n\nContent here.")
        assert markdown_result.is_valid()
        assert markdown_result.content_format == ContentFormat.MARKDOWN
        
        # Test HTML parsing
        html_result = parse_html_content("<h1>Test</h1><p>Content</p>")
        assert html_result.is_valid()
        assert html_result.content_format == ContentFormat.HTML
        
        # Test JSON parsing
        json_result = parse_json_content('{"test": "value"}')
        assert json_result.is_valid()
        assert json_result.content_format == ContentFormat.JSON
        assert json_result.content == {"test": "value"}
    
    def test_validate_content_round_trip_function(self):
        """Test the validate_content_round_trip convenience function."""
        # Test with valid markdown
        valid_markdown = "# Title\n\nContent here."
        assert validate_content_round_trip(valid_markdown, ContentFormat.MARKDOWN) is True
        
        # Test with valid JSON
        valid_json = '{"key": "value", "number": 42}'
        assert validate_content_round_trip(valid_json, ContentFormat.JSON) is True
        
        # Test with invalid content (should return False, not raise exception)
        invalid_json = '{"invalid": json}'
        assert validate_content_round_trip(invalid_json, ContentFormat.JSON) is False


class TestParsedContent:
    """Test cases for the ParsedContent class."""
    
    def test_parsed_content_creation(self):
        """Test creating ParsedContent objects."""
        content = "Test content"
        parsed = ParsedContent(
            content=content,
            content_type=ContentType.TEXT,
            content_format=ContentFormat.PLAIN_TEXT,
            metadata={"test": "value"}
        )
        
        assert parsed.content == content
        assert parsed.content_type == ContentType.TEXT
        assert parsed.content_format == ContentFormat.PLAIN_TEXT
        assert parsed.metadata["test"] == "value"
        assert parsed.is_valid() is True  # Default validation result is success
    
    def test_parsed_content_with_validation_errors(self):
        """Test ParsedContent with validation errors."""
        validation_result = ValidationResult.failure(
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        parsed = ParsedContent(
            content="test",
            content_type=ContentType.TEXT,
            content_format=ContentFormat.PLAIN_TEXT,
            validation_result=validation_result
        )
        
        assert parsed.is_valid() is False
        assert len(parsed.get_errors()) == 2
        assert len(parsed.get_warnings()) == 1
        assert "Error 1" in parsed.get_errors()
        assert "Warning 1" in parsed.get_warnings()


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ContentParser()
    
    def test_validation_error_on_invalid_content(self):
        """Test that ValidationError is raised for invalid content."""
        # Create content that will fail validation
        invalid_markdown = "```\nunclosed code block"
        
        with pytest.raises(ValidationError) as exc_info:
            self.parser.parse_content(invalid_markdown, ContentFormat.MARKDOWN)
        
        assert exc_info.value.status_code == 400
        assert "Content validation failed" in str(exc_info.value)
        assert "errors" in exc_info.value.details
    
    def test_content_parsing_error_details(self):
        """Test ContentParsingError contains proper details."""
        with pytest.raises(ContentParsingError) as exc_info:
            self.parser.parse_content("", ContentFormat.PLAIN_TEXT)
        
        error = exc_info.value
        assert error.status_code == 400
        assert error.error_code == "CONTENT_PARSING_ERROR"
        assert "Content cannot be empty" in error.message
    
    def test_unexpected_error_handling(self):
        """Test handling of unexpected errors during parsing."""
        # Mock the format parsers dictionary to raise an unexpected error
        original_parsers = self.parser._format_parsers.copy()
        
        def mock_parser(content):
            raise RuntimeError("Unexpected error")
        
        self.parser._format_parsers[ContentFormat.PLAIN_TEXT] = mock_parser
        
        try:
            with pytest.raises(ContentParsingError) as exc_info:
                self.parser.parse_content("test", ContentFormat.PLAIN_TEXT, validate=False)
            
            assert "Unexpected error during parsing" in str(exc_info.value)
        finally:
            # Restore original parsers
            self.parser._format_parsers = original_parsers


class TestHelperMethods:
    """Test cases for helper methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ContentParser()
    
    def test_calculate_json_depth(self):
        """Test JSON depth calculation."""
        # Simple object (depth 1 - has one level of keys)
        simple = {"key": "value"}
        assert self.parser._calculate_json_depth(simple) == 1
        
        # Nested object (depth 3 - root -> level1 -> level2 -> level3)
        nested = {"level1": {"level2": {"level3": "value"}}}
        assert self.parser._calculate_json_depth(nested) == 3
        
        # Array with nested objects (depth 3 - root -> array -> object -> nested)
        array_nested = [{"nested": {"deep": "value"}}]
        assert self.parser._calculate_json_depth(array_nested) == 3
        
        # Empty structures
        assert self.parser._calculate_json_depth({}) == 0
        assert self.parser._calculate_json_depth([]) == 0
    
    def test_has_nested_objects(self):
        """Test nested object detection."""
        # Simple object
        simple = {"key": "value", "number": 42}
        assert self.parser._has_nested_objects(simple) is False
        
        # Object with nested dict
        nested_dict = {"key": "value", "nested": {"inner": "value"}}
        assert self.parser._has_nested_objects(nested_dict) is True
        
        # Object with nested array
        nested_array = {"key": "value", "array": [1, 2, 3]}
        assert self.parser._has_nested_objects(nested_array) is True
        
        # Array with nested objects
        array_with_objects = [{"key": "value"}, "simple"]
        assert self.parser._has_nested_objects(array_with_objects) is True
        
        # Simple array
        simple_array = [1, 2, 3, "string"]
        assert self.parser._has_nested_objects(simple_array) is False
    
    def test_calculate_xml_depth(self):
        """Test XML depth calculation."""
        # Simple element
        simple = ET.fromstring("<root>content</root>")
        assert self.parser._calculate_xml_depth(simple) == 0
        
        # Nested elements
        nested_xml = """<root>
            <level1>
                <level2>
                    <level3>content</level3>
                </level2>
            </level1>
        </root>"""
        nested = ET.fromstring(nested_xml)
        assert self.parser._calculate_xml_depth(nested) == 3
    
    def test_is_valid_url(self):
        """Test URL validation."""
        # Valid URLs
        assert self.parser._is_valid_url("https://example.com") is True
        assert self.parser._is_valid_url("http://test.org/path") is True
        assert self.parser._is_valid_url("ftp://files.example.com") is True
        
        # Invalid URLs
        assert self.parser._is_valid_url("not-a-url") is False
        assert self.parser._is_valid_url("://missing-scheme") is False
        assert self.parser._is_valid_url("https://") is False
        assert self.parser._is_valid_url("") is False


if __name__ == "__main__":
    pytest.main([__file__])