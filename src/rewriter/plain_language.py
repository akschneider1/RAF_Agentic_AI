"""
Plain Language Rewriter - Transform legalese into clear, accessible text

Addresses RAF RFI Requirement #3:
"Tools for rewriting regulations and/or policy guidance documents to eliminate
legalese, verbosity and vagueness while retaining core requirements"
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ReadabilityLevel(Enum):
    """Target readability levels"""
    GENERAL_PUBLIC = "general_public"  # 8th grade reading level
    BUSINESS = "business"  # Professional but accessible
    TECHNICAL = "technical"  # Maintains technical precision
    LEGAL = "legal"  # Simplified but legally precise


@dataclass
class RewriteResult:
    """Result of a plain language rewrite"""
    original_text: str
    rewritten_text: str
    changes_made: list[dict[str, str]]
    readability_before: float
    readability_after: float
    confidence_score: float  # Confidence legal meaning preserved
    warnings: list[str] = field(default_factory=list)

    def get_comparison(self) -> str:
        """Generate side-by-side comparison"""
        lines = [
            "## Original vs. Rewritten",
            "",
            "### Original",
            self.original_text,
            "",
            "### Plain Language Version",
            self.rewritten_text,
            "",
            f"**Readability improvement:** {self.readability_before:.1f} → {self.readability_after:.1f}",
            f"**Legal meaning confidence:** {self.confidence_score:.0%}",
        ]

        if self.changes_made:
            lines.extend(["", "### Key Changes"])
            for change in self.changes_made[:10]:
                lines.append(f"- \"{change.get('before', '')}\" → \"{change.get('after', '')}\"")

        if self.warnings:
            lines.extend(["", "### Warnings"])
            for warning in self.warnings:
                lines.append(f"- ⚠️ {warning}")

        return "\n".join(lines)


class PlainLanguageRewriter:
    """
    Rewrites legal/regulatory text in plain language while preserving legal meaning.

    Key principles applied:
    1. Use active voice
    2. Use short sentences (avg 15-20 words)
    3. Use common words instead of jargon
    4. Use "you" and "we" for clarity
    5. Use clear structure with headings
    6. Define technical terms when needed
    """

    # Common legalese replacements
    JARGON_MAP = {
        r"\bherein\b": "in this document",
        r"\bhereof\b": "of this document",
        r"\bhereby\b": "",
        r"\bhereto\b": "to this",
        r"\bhereafter\b": "after this",
        r"\bheretofore\b": "before this",
        r"\bthereof\b": "of that",
        r"\btherein\b": "in that",
        r"\bthereto\b": "to that",
        r"\bwhereof\b": "of which",
        r"\bwherein\b": "in which",
        r"\bwhereas\b": "",  # Usually just ceremonial
        r"\baforementioned\b": "mentioned above",
        r"\baforesaid\b": "mentioned above",
        r"\bnotwithstanding\b": "despite",
        r"\bpursuant to\b": "under",
        r"\bin accordance with\b": "following",
        r"\bwith respect to\b": "about",
        r"\bin the event that\b": "if",
        r"\bfor the purpose of\b": "to",
        r"\bin order to\b": "to",
        r"\bprior to\b": "before",
        r"\bsubsequent to\b": "after",
        r"\bat such time as\b": "when",
        r"\bin the amount of\b": "for",
        r"\bby means of\b": "by",
        r"\bby virtue of\b": "by",
        r"\bdue to the fact that\b": "because",
        r"\bin lieu of\b": "instead of",
        r"\bwith regard to\b": "about",
        r"\bit is necessary that\b": "must",
        r"\bis authorized to\b": "may",
        r"\bis required to\b": "must",
        r"\bis entitled to\b": "may",
        r"\bshall be\b": "is" if "shall be" else "will be",
        r"\bshall not\b": "must not",
        r"\bshall have\b": "has",
    }

    # Passive voice patterns to flag
    PASSIVE_PATTERNS = [
        r"\b(is|are|was|were|be|been|being)\s+(\w+ed)\b",
        r"\b(is|are|was|were)\s+to\s+be\s+(\w+ed)\b",
    ]

    SYSTEM_PROMPT = """You are an expert in plain language writing for government documents.
Your task is to rewrite legal and regulatory text to be clear and accessible while
preserving all legal requirements and obligations.

Guidelines:
1. Use active voice (e.g., "You must submit" not "The form must be submitted")
2. Use short sentences (15-20 words average)
3. Use common words (e.g., "use" not "utilize", "help" not "facilitate")
4. Use "you" to address the reader and "we" for the agency
5. Put the most important information first
6. Use bullet points or numbered lists for steps or requirements
7. Define technical terms if they must be used
8. Maintain legal precision - don't change what's required, just how it's expressed

When rewriting:
- PRESERVE all legal requirements, deadlines, and obligations
- PRESERVE all references to specific sections, forms, or agencies
- PRESERVE technical terms that have specific legal meaning (define them if needed)
- CONVERT passive voice to active voice
- BREAK long sentences into shorter ones
- REPLACE jargon with plain language equivalents
- ADD structure (headings, lists) where helpful

Format your response as:
REWRITTEN TEXT:
[Your plain language version]

CHANGES MADE:
- [Brief description of key changes]

LEGAL CONFIDENCE: [0-100]%
[Brief explanation of confidence level]

WARNINGS:
- [Any terms or provisions where meaning might be affected]"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client"""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client

    def rewrite(
        self,
        text: str,
        level: ReadabilityLevel = ReadabilityLevel.GENERAL_PUBLIC,
        preserve_structure: bool = False,
    ) -> RewriteResult:
        """
        Rewrite text in plain language.

        Args:
            text: The legal/regulatory text to rewrite
            level: Target readability level
            preserve_structure: If True, maintain original paragraph structure

        Returns:
            RewriteResult with original, rewritten text, and analysis
        """
        # Calculate initial readability
        readability_before = self._calculate_readability(text)

        # Phase 1: Quick pattern-based cleanup
        cleaned_text = self._apply_jargon_replacements(text)

        # Phase 2: LLM-powered rewrite
        if self.api_key:
            result = self._llm_rewrite(cleaned_text, level, preserve_structure)
            rewritten = result.get("rewritten", cleaned_text)
            changes = result.get("changes", [])
            confidence = result.get("confidence", 0.8)
            warnings = result.get("warnings", [])
        else:
            rewritten = cleaned_text
            changes = self._list_jargon_changes(text, cleaned_text)
            confidence = 0.95  # Pattern-based is conservative
            warnings = ["LLM rewrite unavailable - pattern-based cleanup only"]

        # Calculate final readability
        readability_after = self._calculate_readability(rewritten)

        return RewriteResult(
            original_text=text,
            rewritten_text=rewritten,
            changes_made=changes,
            readability_before=readability_before,
            readability_after=readability_after,
            confidence_score=confidence,
            warnings=warnings,
        )

    def _apply_jargon_replacements(self, text: str) -> str:
        """Apply pattern-based jargon replacements"""
        result = text

        for pattern, replacement in self.JARGON_MAP.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Clean up double spaces
        result = re.sub(r"  +", " ", result)

        return result

    def _llm_rewrite(
        self,
        text: str,
        level: ReadabilityLevel,
        preserve_structure: bool,
    ) -> dict[str, Any]:
        """Use LLM for sophisticated rewriting"""

        level_instruction = {
            ReadabilityLevel.GENERAL_PUBLIC: "Write at an 8th grade reading level. Anyone should understand this.",
            ReadabilityLevel.BUSINESS: "Write for business professionals. Clear but can include common business terms.",
            ReadabilityLevel.TECHNICAL: "Maintain technical precision. Simplify language but keep necessary technical terms.",
            ReadabilityLevel.LEGAL: "Simplify while maintaining legal precision. Define technical terms.",
        }

        prompt = f"""Rewrite the following regulatory/legal text in plain language.

Target audience: {level_instruction.get(level, level_instruction[ReadabilityLevel.GENERAL_PUBLIC])}

{"Preserve the original paragraph structure." if preserve_structure else "Feel free to restructure for clarity."}

TEXT TO REWRITE:
{text}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            return self._parse_llm_response(content)

        except Exception as e:
            return {
                "rewritten": text,
                "changes": [],
                "confidence": 0.5,
                "warnings": [f"LLM rewrite failed: {str(e)}"],
            }

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse structured LLM response"""
        result = {
            "rewritten": "",
            "changes": [],
            "confidence": 0.8,
            "warnings": [],
        }

        # Extract rewritten text
        if "REWRITTEN TEXT:" in response:
            parts = response.split("REWRITTEN TEXT:", 1)
            if len(parts) > 1:
                rest = parts[1]
                # Find the end (next section or end of text)
                for marker in ["CHANGES MADE:", "LEGAL CONFIDENCE:", "WARNINGS:"]:
                    if marker in rest:
                        rest = rest.split(marker)[0]
                        break
                result["rewritten"] = rest.strip()

        # Extract changes
        if "CHANGES MADE:" in response:
            changes_section = response.split("CHANGES MADE:", 1)[1]
            for marker in ["LEGAL CONFIDENCE:", "WARNINGS:"]:
                if marker in changes_section:
                    changes_section = changes_section.split(marker)[0]
                    break

            for line in changes_section.strip().split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    result["changes"].append({"description": line[1:].strip()})

        # Extract confidence
        if "LEGAL CONFIDENCE:" in response:
            conf_section = response.split("LEGAL CONFIDENCE:", 1)[1]
            match = re.search(r"(\d+)\s*%", conf_section)
            if match:
                result["confidence"] = int(match.group(1)) / 100

        # Extract warnings
        if "WARNINGS:" in response:
            warnings_section = response.split("WARNINGS:", 1)[1]
            for line in warnings_section.strip().split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    result["warnings"].append(line[1:].strip())

        return result

    def _list_jargon_changes(self, original: str, cleaned: str) -> list[dict[str, str]]:
        """List what jargon was replaced"""
        changes = []

        for pattern, replacement in self.JARGON_MAP.items():
            matches = re.findall(pattern, original, re.IGNORECASE)
            if matches:
                for match in set(matches):
                    changes.append({
                        "before": match,
                        "after": replacement if replacement else "(removed)",
                    })

        return changes

    def _calculate_readability(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid readability score.

        Higher score = easier to read
        90-100: 5th grade
        80-89: 6th grade
        70-79: 7th grade
        60-69: 8th-9th grade
        50-59: 10th-12th grade
        30-49: College
        0-29: College graduate
        """
        # Count sentences
        sentences = len(re.findall(r"[.!?]+", text)) or 1

        # Count words
        words = text.split()
        word_count = len(words) or 1

        # Count syllables (simplified)
        syllable_count = 0
        for word in words:
            syllable_count += self._count_syllables(word)

        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (word_count / sentences) - 84.6 * (syllable_count / word_count)

        return max(0, min(100, score))

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word"""
        word = word.lower().strip(".,!?;:")
        if not word:
            return 0

        # Simple heuristic
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e"):
            count -= 1

        return max(1, count)

    def analyze_complexity(self, text: str) -> dict[str, Any]:
        """Analyze text complexity without rewriting"""
        words = text.split()
        sentences = len(re.findall(r"[.!?]+", text)) or 1

        # Find passive voice instances
        passive_instances = []
        for pattern in self.PASSIVE_PATTERNS:
            passive_instances.extend(re.findall(pattern, text, re.IGNORECASE))

        # Find jargon
        jargon_found = []
        for pattern in self.JARGON_MAP.keys():
            matches = re.findall(pattern, text, re.IGNORECASE)
            jargon_found.extend(matches)

        # Long sentences (>30 words)
        long_sentences = []
        for sentence in re.split(r"[.!?]+", text):
            if len(sentence.split()) > 30:
                long_sentences.append(sentence.strip()[:100] + "...")

        return {
            "readability_score": self._calculate_readability(text),
            "word_count": len(words),
            "sentence_count": sentences,
            "avg_sentence_length": len(words) / sentences,
            "passive_voice_count": len(passive_instances),
            "jargon_count": len(jargon_found),
            "jargon_examples": list(set(jargon_found))[:10],
            "long_sentences": long_sentences[:5],
            "complexity_rating": self._rate_complexity(text),
        }

    def _rate_complexity(self, text: str) -> str:
        """Rate overall complexity"""
        score = self._calculate_readability(text)

        if score >= 70:
            return "Accessible"
        elif score >= 50:
            return "Moderate"
        elif score >= 30:
            return "Complex"
        else:
            return "Very Complex"
