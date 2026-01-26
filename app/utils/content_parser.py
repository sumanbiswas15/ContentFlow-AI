"""
Content parsing and validation system for ContentFlow AI.

This module provides the ContentParser class with format-specific parsing methods,
content validation logic, and comprehensive error handling for parsing failures.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import markdown
from bs4 import BeautifulSoup

from app.models.base import ContentFormat, ContentType, ValidationResult
from app.core.exceptions import ContentParsingError, ValidationError


class ParsedContent:
    """Represents parsed content with metadata and validation results."""
    
    def __init__(
        self,
        content: Union[str, Dict[str, Any]],
        content_type: ContentType,
        content_format: ContentFormat,
        metadata: Optional[Dict[str, Any]] = None,
        validation_result: Optional[ValidationResult] = None
    ):
        self.content = content
        self.content_type = content_type
        self.content_format = content_format
        self.metadata = metadata or {}
        self.validation_result = validation_result or ValidationResult.success()
        
    def is_valid(self) -> bool:
        """Check if the parsed content is valid."""
        return self.validation_result.is_valid
    
    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.validation_result.errors
    
    def get_warnings(self) -> List[str]:
        """Get validation warnings."""
        return self.validation_result.warnings


class ContentParser:
    """
    Content parsing and validation system with format-specific parsing methods.
    
    Supports parsing of various content formats including markdown, HTML, plain text,
    JSON, and XML with comprehensive validation and error handling.
    """
    
    def __init__(self):
        """Initialize the content parser."""
        self._format_parsers = {
            ContentFormat.MARKDOWN: self._parse_markdown,
            ContentFormat.HTML: self._parse_html,
            ContentFormat.PLAIN_TEXT: self._parse_plain_text,
            ContentFormat.JSON: self._parse_json,
            ContentFormat.XML: self._parse_xml
        }
        
        self._format_validators = {
            ContentFormat.MARKDOWN: self._validate_markdown,
            ContentFormat.HTML: self._validate_html,
            ContentFormat.PLAIN_TEXT: self._validate_plain_text,
            ContentFormat.JSON: self._validate_json,
            ContentFormat.XML: self._validate_xml
        }
    
    def parse_content(
        self,
        raw_content: str,
        content_format: ContentFormat,
        content_type: ContentType = ContentType.TEXT,
        validate: bool = True
    ) -> ParsedContent:
        """
        Parse content according to format specifications.
        
        Args:
            raw_content: The raw content string to parse
            content_format: The format of the content
            content_type: The type of content (default: TEXT)
            validate: Whether to validate the content (default: True)
            
        Returns:
            ParsedContent object with parsed content and validation results
            
        Raises:
            ContentParsingError: When parsing fails
            ValidationError: When validation fails
        """
        if not raw_content:
            raise ContentParsingError(
                content_type=content_type.value,
                message="Content cannot be empty"
            )
        
        if content_format not in self._format_parsers:
            raise ContentParsingError(
                content_type=content_type.value,
                message=f"Unsupported content format: {content_format}"
            )
        
        try:
            # Parse the content
            parser = self._format_parsers[content_format]
            parsed_content, metadata = parser(raw_content)
            
            # Create parsed content object
            parsed = ParsedContent(
                content=parsed_content,
                content_type=content_type,
                content_format=content_format,
                metadata=metadata
            )
            
            # Validate if requested
            if validate:
                validation_result = self.validate_content(parsed)
                parsed.validation_result = validation_result
                
                if not validation_result.is_valid:
                    raise ValidationError(
                        message=f"Content validation failed: {'; '.join(validation_result.errors)}",
                        details={
                            "errors": validation_result.errors,
                            "warnings": validation_result.warnings,
                            "content_format": content_format.value,
                            "content_type": content_type.value
                        }
                    )
            
            return parsed
            
        except (json.JSONDecodeError, ET.ParseError, UnicodeDecodeError) as e:
            raise ContentParsingError(
                content_type=content_type.value,
                message=f"Failed to parse {content_format.value} content: {str(e)}"
            )
        except Exception as e:
            if isinstance(e, (ContentParsingError, ValidationError)):
                raise
            raise ContentParsingError(
                content_type=content_type.value,
                message=f"Unexpected error during parsing: {str(e)}"
            )
    
    def validate_content(self, parsed_content: ParsedContent) -> ValidationResult:
        """
        Validate parsed content for format compliance and data integrity.
        
        Args:
            parsed_content: The parsed content to validate
            
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        content_format = parsed_content.content_format
        
        if content_format not in self._format_validators:
            return ValidationResult.failure([f"No validator available for format: {content_format}"])
        
        try:
            validator = self._format_validators[content_format]
            return validator(parsed_content)
        except Exception as e:
            return ValidationResult.failure([f"Validation error: {str(e)}"])
    
    def format_content(
        self,
        parsed_content: ParsedContent,
        target_format: Optional[ContentFormat] = None
    ) -> str:
        """
        Format parsed content back into valid output format.
        
        Args:
            parsed_content: The parsed content to format
            target_format: Target format (defaults to original format)
            
        Returns:
            Formatted content string
            
        Raises:
            ContentParsingError: When formatting fails
        """
        format_to_use = target_format or parsed_content.content_format
        
        try:
            if format_to_use == ContentFormat.MARKDOWN:
                return self._format_to_markdown(parsed_content)
            elif format_to_use == ContentFormat.HTML:
                return self._format_to_html(parsed_content)
            elif format_to_use == ContentFormat.PLAIN_TEXT:
                return self._format_to_plain_text(parsed_content)
            elif format_to_use == ContentFormat.JSON:
                return self._format_to_json(parsed_content)
            elif format_to_use == ContentFormat.XML:
                return self._format_to_xml(parsed_content)
            else:
                raise ContentParsingError(
                    content_type=parsed_content.content_type.value,
                    message=f"Unsupported target format: {format_to_use}"
                )
        except Exception as e:
            if isinstance(e, ContentParsingError):
                raise
            raise ContentParsingError(
                content_type=parsed_content.content_type.value,
                message=f"Failed to format content to {format_to_use}: {str(e)}"
            )
    
    # Format-specific parsing methods
    
    def _parse_markdown(self, content: str) -> tuple[str, Dict[str, Any]]:
        """Parse markdown content."""
        # Convert markdown to HTML for processing
        html_content = markdown.markdown(content, extensions=['meta', 'tables', 'fenced_code'])
        
        # Extract metadata if present
        md = markdown.Markdown(extensions=['meta'])
        md.convert(content)
        metadata = {
            'markdown_meta': getattr(md, 'Meta', {}),
            'has_tables': '|' in content and '---' in content,
            'has_code_blocks': '```' in content or '    ' in content,
            'has_links': '[' in content and '](' in content,
            'line_count': len(content.split('\n')),
            'word_count': len(content.split())
        }
        
        return content, metadata
    
    def _parse_html(self, content: str) -> tuple[str, Dict[str, Any]]:
        """Parse HTML content."""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract metadata
        metadata = {
            'title': soup.title.string if soup.title else None,
            'meta_tags': {tag.get('name', tag.get('property', 'unknown')): tag.get('content', '') 
                         for tag in soup.find_all('meta') if tag.get('content')},
            'has_images': len(soup.find_all('img')) > 0,
            'has_links': len(soup.find_all('a')) > 0,
            'has_forms': len(soup.find_all('form')) > 0,
            'tag_count': len(soup.find_all()),
            'text_content': soup.get_text().strip()
        }
        
        return content, metadata
    
    def _parse_plain_text(self, content: str) -> tuple[str, Dict[str, Any]]:
        """Parse plain text content."""
        lines = content.split('\n')
        
        metadata = {
            'line_count': len(lines),
            'word_count': len(content.split()),
            'character_count': len(content),
            'paragraph_count': len([line for line in lines if line.strip()]),
            'has_urls': bool(re.search(r'https?://\S+', content)),
            'has_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)),
            'encoding': 'utf-8'  # Assume UTF-8 for now
        }
        
        return content, metadata
    
    def _parse_json(self, content: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse JSON content."""
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError as e:
            raise ContentParsingError(
                content_type="json",
                message=f"Invalid JSON format: {str(e)}"
            )
        
        metadata = {
            'json_type': type(parsed_json).__name__,
            'key_count': len(parsed_json) if isinstance(parsed_json, dict) else None,
            'item_count': len(parsed_json) if isinstance(parsed_json, list) else None,
            'max_depth': self._calculate_json_depth(parsed_json),
            'has_nested_objects': self._has_nested_objects(parsed_json),
            'size_bytes': len(content.encode('utf-8'))
        }
        
        return parsed_json, metadata
    
    def _parse_xml(self, content: str) -> tuple[ET.Element, Dict[str, Any]]:
        """Parse XML content."""
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            raise ContentParsingError(
                content_type="xml",
                message=f"Invalid XML format: {str(e)}"
            )
        
        metadata = {
            'root_tag': root.tag,
            'namespace': root.tag.split('}')[0][1:] if root.tag.startswith('{') else None,
            'element_count': len(list(root.iter())),
            'attribute_count': sum(len(elem.attrib) for elem in root.iter()),
            'max_depth': self._calculate_xml_depth(root),
            'has_text_content': any(elem.text and elem.text.strip() for elem in root.iter()),
            'size_bytes': len(content.encode('utf-8'))
        }
        
        return root, metadata
    
    # Format-specific validation methods
    
    def _validate_markdown(self, parsed_content: ParsedContent) -> ValidationResult:
        """Validate markdown content."""
        result = ValidationResult.success()
        content = parsed_content.content
        metadata = parsed_content.metadata
        
        # Check for common markdown issues
        if not isinstance(content, str):
            result.add_error("Markdown content must be a string")
            return result
        
        # Check for malformed links
        link_pattern = r'\[([^\]]*)\]\(([^)]*)\)'
        links = re.findall(link_pattern, content)
        for link_text, link_url in links:
            if not link_url.strip():
                result.add_warning(f"Empty URL in link: [{link_text}]()")
            elif not self._is_valid_url(link_url) and not link_url.startswith('#'):
                result.add_warning(f"Potentially invalid URL: {link_url}")
        
        # Check for unmatched code blocks
        code_block_count = content.count('```')
        if code_block_count % 2 != 0:
            result.add_error("Unmatched code block delimiters (```)")
        
        # Check for very long lines (readability)
        lines = content.split('\n')
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            result.add_warning(f"Lines longer than 120 characters: {long_lines[:5]}")
        
        return result
    
    def _validate_html(self, parsed_content: ParsedContent) -> ValidationResult:
        """Validate HTML content."""
        result = ValidationResult.success()
        content = parsed_content.content
        
        if not isinstance(content, str):
            result.add_error("HTML content must be a string")
            return result
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Check for unclosed tags
            if soup.find_all(string=lambda text: '<' in str(text) and '>' in str(text)):
                result.add_warning("Potential unclosed or malformed tags detected")
            
            # Check for missing alt attributes on images
            images_without_alt = soup.find_all('img', alt=False)
            if images_without_alt:
                result.add_warning(f"Found {len(images_without_alt)} images without alt attributes")
            
            # Check for empty links
            empty_links = soup.find_all('a', href=False)
            if empty_links:
                result.add_warning(f"Found {len(empty_links)} links without href attributes")
            
            # Check for inline styles (accessibility concern)
            elements_with_style = soup.find_all(style=True)
            if elements_with_style:
                result.add_warning(f"Found {len(elements_with_style)} elements with inline styles")
                
        except Exception as e:
            result.add_error(f"HTML parsing error: {str(e)}")
        
        return result
    
    def _validate_plain_text(self, parsed_content: ParsedContent) -> ValidationResult:
        """Validate plain text content."""
        result = ValidationResult.success()
        content = parsed_content.content
        
        if not isinstance(content, str):
            result.add_error("Plain text content must be a string")
            return result
        
        # Check for control characters
        control_chars = [char for char in content if ord(char) < 32 and char not in '\n\r\t']
        if control_chars:
            result.add_warning(f"Found {len(control_chars)} control characters")
        
        # Check encoding issues
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            result.add_error("Content contains characters that cannot be encoded as UTF-8")
        
        # Check for extremely long lines
        lines = content.split('\n')
        very_long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 1000]
        if very_long_lines:
            result.add_warning(f"Very long lines detected: {very_long_lines[:3]}")
        
        return result
    
    def _validate_json(self, parsed_content: ParsedContent) -> ValidationResult:
        """Validate JSON content."""
        result = ValidationResult.success()
        content = parsed_content.content
        
        if not isinstance(content, (dict, list, str, int, float, bool)) and content is not None:
            result.add_error("Invalid JSON data type")
            return result
        
        # Check for circular references (basic check)
        try:
            json.dumps(content)
        except (TypeError, ValueError) as e:
            result.add_error(f"JSON serialization error: {str(e)}")
        
        # Check depth
        max_depth = parsed_content.metadata.get('max_depth', 0)
        if max_depth > 20:
            result.add_warning(f"JSON structure is very deep ({max_depth} levels)")
        
        # Check size
        size_bytes = parsed_content.metadata.get('size_bytes', 0)
        if size_bytes > 1024 * 1024:  # 1MB
            result.add_warning(f"JSON content is large ({size_bytes} bytes)")
        
        return result
    
    def _validate_xml(self, parsed_content: ParsedContent) -> ValidationResult:
        """Validate XML content."""
        result = ValidationResult.success()
        content = parsed_content.content
        
        if not isinstance(content, ET.Element):
            result.add_error("XML content must be an Element object")
            return result
        
        # Check for namespace issues
        namespace = parsed_content.metadata.get('namespace')
        if namespace and not self._is_valid_url(namespace):
            result.add_warning(f"Namespace URI may be invalid: {namespace}")
        
        # Check depth
        max_depth = parsed_content.metadata.get('max_depth', 0)
        if max_depth > 15:
            result.add_warning(f"XML structure is very deep ({max_depth} levels)")
        
        # Check for empty elements
        empty_elements = [elem for elem in content.iter() if not elem.text and not elem.tail and not elem.attrib and len(elem) == 0]
        if empty_elements:
            result.add_warning(f"Found {len(empty_elements)} empty elements")
        
        return result
    
    # Formatting methods
    
    def _format_to_markdown(self, parsed_content: ParsedContent) -> str:
        """Format content to markdown."""
        if parsed_content.content_format == ContentFormat.MARKDOWN:
            return str(parsed_content.content)
        elif parsed_content.content_format == ContentFormat.HTML:
            # Basic HTML to Markdown conversion
            soup = BeautifulSoup(parsed_content.content, 'html.parser')
            return soup.get_text()
        elif parsed_content.content_format == ContentFormat.PLAIN_TEXT:
            return str(parsed_content.content)
        else:
            return str(parsed_content.content)
    
    def _format_to_html(self, parsed_content: ParsedContent) -> str:
        """Format content to HTML."""
        if parsed_content.content_format == ContentFormat.HTML:
            return str(parsed_content.content)
        elif parsed_content.content_format == ContentFormat.MARKDOWN:
            return markdown.markdown(str(parsed_content.content))
        elif parsed_content.content_format == ContentFormat.PLAIN_TEXT:
            # Escape HTML and convert newlines to <br>
            import html
            escaped = html.escape(str(parsed_content.content))
            return escaped.replace('\n', '<br>\n')
        else:
            import html
            return html.escape(str(parsed_content.content))
    
    def _format_to_plain_text(self, parsed_content: ParsedContent) -> str:
        """Format content to plain text."""
        if parsed_content.content_format == ContentFormat.PLAIN_TEXT:
            return str(parsed_content.content)
        elif parsed_content.content_format == ContentFormat.HTML:
            soup = BeautifulSoup(parsed_content.content, 'html.parser')
            return soup.get_text()
        elif parsed_content.content_format == ContentFormat.MARKDOWN:
            # Convert markdown to HTML first, then extract text
            html_content = markdown.markdown(str(parsed_content.content))
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
        else:
            return str(parsed_content.content)
    
    def _format_to_json(self, parsed_content: ParsedContent) -> str:
        """Format content to JSON."""
        if parsed_content.content_format == ContentFormat.JSON:
            return json.dumps(parsed_content.content, indent=2, ensure_ascii=False)
        else:
            # Wrap other content types in a JSON structure
            return json.dumps({
                "content": str(parsed_content.content),
                "content_type": parsed_content.content_type.value,
                "content_format": parsed_content.content_format.value,
                "metadata": parsed_content.metadata
            }, indent=2, ensure_ascii=False)
    
    def _format_to_xml(self, parsed_content: ParsedContent) -> str:
        """Format content to XML."""
        if parsed_content.content_format == ContentFormat.XML:
            return ET.tostring(parsed_content.content, encoding='unicode')
        else:
            # Create a simple XML wrapper
            root = ET.Element("content")
            root.set("type", parsed_content.content_type.value)
            root.set("format", parsed_content.content_format.value)
            root.text = str(parsed_content.content)
            return ET.tostring(root, encoding='unicode')
    
    # Helper methods
    
    def _calculate_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of a JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_json_depth(value, current_depth + 1) for value in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _has_nested_objects(self, obj: Any) -> bool:
        """Check if JSON object has nested objects or arrays."""
        if isinstance(obj, dict):
            return any(isinstance(value, (dict, list)) for value in obj.values())
        elif isinstance(obj, list):
            return any(isinstance(item, (dict, list)) for item in obj)
        return False
    
    def _calculate_xml_depth(self, element: ET.Element, current_depth: int = 0) -> int:
        """Calculate the maximum depth of an XML element."""
        if not list(element):
            return current_depth
        return max(self._calculate_xml_depth(child, current_depth + 1) for child in element)
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


# Convenience functions for common use cases

def parse_markdown_content(content: str, validate: bool = True) -> ParsedContent:
    """Parse markdown content with validation."""
    parser = ContentParser()
    return parser.parse_content(content, ContentFormat.MARKDOWN, ContentType.TEXT, validate)


def parse_html_content(content: str, validate: bool = True) -> ParsedContent:
    """Parse HTML content with validation."""
    parser = ContentParser()
    return parser.parse_content(content, ContentFormat.HTML, ContentType.TEXT, validate)


def parse_json_content(content: str, validate: bool = True) -> ParsedContent:
    """Parse JSON content with validation."""
    parser = ContentParser()
    return parser.parse_content(content, ContentFormat.JSON, ContentType.TEXT, validate)


def validate_content_round_trip(content: str, content_format: ContentFormat) -> bool:
    """
    Test round-trip consistency: parse -> format -> parse should produce equivalent objects.
    
    Args:
        content: The content to test
        content_format: The format of the content
        
    Returns:
        True if round-trip is consistent, False otherwise
    """
    try:
        parser = ContentParser()
        
        # First parse
        parsed1 = parser.parse_content(content, content_format, validate=False)
        
        # Format back to string
        formatted = parser.format_content(parsed1)
        
        # Parse again
        parsed2 = parser.parse_content(formatted, content_format, validate=False)
        
        # Compare content (handling different types appropriately)
        if content_format == ContentFormat.JSON:
            return parsed1.content == parsed2.content
        elif content_format == ContentFormat.XML:
            # For XML, compare string representations
            return ET.tostring(parsed1.content) == ET.tostring(parsed2.content)
        else:
            # For text-based formats, compare strings
            return str(parsed1.content).strip() == str(parsed2.content).strip()
            
    except Exception:
        return False