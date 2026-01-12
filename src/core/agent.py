"""
Agentic Engine - Core AI agent for document analysis and reform recommendations

This implements an agentic approach inspired by The Agentic State framework,
where the AI system can reason about regulatory burden and propose reforms.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .burden_taxonomy import BurdenCategory, BurdenTaxonomy, BurdenType, IdentifiedBurden, Severity
from .document import DocumentType, LegalDocument, Section


class AnalysisMode(Enum):
    """Modes of analysis the agent can perform"""
    BURDEN_SCAN = "burden_scan"  # Identify procedural burdens
    GAP_ANALYSIS = "gap_analysis"  # Find gaps between statute and regulation
    PLAIN_LANGUAGE = "plain_language"  # Rewrite for clarity
    CROSS_STATE = "cross_state"  # Compare across jurisdictions
    REFORM_DRAFT = "reform_draft"  # Draft reform language


@dataclass
class AnalysisResult:
    """Result of an analysis operation"""
    mode: AnalysisMode
    document_title: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    citations: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    """Current state of the agentic analysis"""
    documents_analyzed: int = 0
    burdens_identified: int = 0
    reforms_proposed: int = 0
    current_task: Optional[str] = None
    memory: list[dict[str, Any]] = field(default_factory=list)


class AgenticEngine:
    """
    Core agentic engine for analyzing legal documents and proposing reforms.

    This engine combines:
    1. Pattern-based burden detection (fast, explainable)
    2. LLM-powered deep analysis (nuanced, context-aware)
    3. Agentic reasoning for reform recommendations

    Designed to support the Recoding America Fund's deproceduralization goals.
    """

    SYSTEM_PROMPT = """You are an expert legal analyst specializing in regulatory reform
and administrative burden reduction. Your role is to:

1. Identify procedural burdens in government statutes, regulations, and policies
2. Analyze whether regulations exceed their statutory authority
3. Propose plain-language rewrites that maintain legal precision
4. Compare requirements across jurisdictions to identify best practices
5. Draft reform language that eliminates unnecessary burden

When analyzing text, always:
- Cite specific sections and provisions
- Explain WHY something is burdensome (time, cost, accessibility)
- Consider impacts on different populations (small businesses, elderly, rural)
- Propose concrete, actionable reforms
- Maintain legal validity of any proposed changes

Be direct and specific. Government reformers need actionable insights, not vague observations."""

    BURDEN_ANALYSIS_PROMPT = """Analyze the following legal text for procedural burdens.

For each burden found, provide:
1. The specific text creating the burden
2. The type of burden (e.g., wet signature, waiting period, in-person requirement)
3. Severity (LOW, MEDIUM, HIGH, CRITICAL)
4. Who is affected
5. Estimated time/cost impact
6. A specific reform recommendation

TEXT TO ANALYZE:
{text}

Respond in JSON format:
{{
    "burdens": [
        {{
            "text_excerpt": "...",
            "burden_type": "...",
            "severity": "...",
            "affected_parties": "...",
            "estimated_impact": "...",
            "reform_recommendation": "..."
        }}
    ],
    "overall_burden_score": 0-10,
    "summary": "..."
}}"""

    PLAIN_LANGUAGE_PROMPT = """Rewrite the following legal text in plain language.

Requirements:
1. Maintain all legal requirements and obligations
2. Eliminate jargon, legalese, and unnecessary complexity
3. Use active voice and short sentences
4. Define technical terms where necessary
5. Organize with clear headings if helpful

ORIGINAL TEXT:
{text}

Provide:
1. The plain language rewrite
2. A side-by-side comparison of key changes
3. Confidence that legal meaning is preserved (0-100%)"""

    GAP_ANALYSIS_PROMPT = """Compare the following regulation against its statutory authority.

STATUTE (Authorizing Law):
{statute}

REGULATION (Agency Implementation):
{regulation}

Analyze for:
1. Requirements in regulation not authorized by statute ("gold-plating")
2. Stricter interpretations than statute requires
3. Additional fees, timeframes, or procedures beyond statutory minimum
4. Missing provisions that statute requires

Respond in JSON format:
{{
    "gaps": [
        {{
            "type": "exceeds_authority|stricter_than_required|missing_requirement",
            "regulation_text": "...",
            "statute_text": "...",
            "analysis": "...",
            "recommendation": "..."
        }}
    ],
    "alignment_score": 0-100,
    "summary": "..."
}}"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.taxonomy = BurdenTaxonomy()
        self.state = AgentState()
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

    def analyze_for_burdens(
        self,
        document: LegalDocument,
        use_llm: bool = True,
    ) -> AnalysisResult:
        """
        Analyze a document for procedural burdens.

        Combines pattern matching (fast, explainable) with LLM analysis (nuanced).
        """
        self.state.current_task = f"Analyzing {document.title} for burdens"

        all_burdens = []

        for section in document.iter_sections():
            # Phase 1: Pattern-based detection
            pattern_burdens = self._pattern_detect_burdens(section)
            all_burdens.extend(pattern_burdens)

            # Phase 2: LLM-powered deep analysis (if enabled and text is substantial)
            if use_llm and section.word_count > 50:
                llm_burdens = self._llm_analyze_section(section)
                all_burdens.extend(llm_burdens)

        # Deduplicate and merge findings
        merged_burdens = self._merge_burden_findings(all_burdens)

        self.state.documents_analyzed += 1
        self.state.burdens_identified += len(merged_burdens)

        return AnalysisResult(
            mode=AnalysisMode.BURDEN_SCAN,
            document_title=document.title,
            findings=[self._burden_to_dict(b) for b in merged_burdens],
            summary=self._generate_burden_summary(merged_burdens),
            recommendations=self._generate_recommendations(merged_burdens),
            confidence_score=0.85,
            citations=[str(s.citation) for s in document.sections if s.citation],
        )

    def _pattern_detect_burdens(self, section: Section) -> list[IdentifiedBurden]:
        """Fast pattern-based burden detection"""
        import re

        burdens = []
        text = section.text

        for burden_type, indicator in self.taxonomy.indicators.items():
            for pattern in indicator.patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    burdens.append(IdentifiedBurden(
                        burden_type=burden_type,
                        category=self.taxonomy.categorize_burden(burden_type),
                        severity=indicator.typical_severity,
                        location=str(section.citation) if section.citation else section.id,
                        text_excerpt=self._get_context(text, match.start(), match.end()),
                        explanation=f"Pattern match: {burden_type.value}",
                    ))

        return burdens

    def _llm_analyze_section(self, section: Section) -> list[IdentifiedBurden]:
        """Deep LLM-based burden analysis"""
        if not self.api_key:
            return []  # Skip if no API key

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=self.SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": self.BURDEN_ANALYSIS_PROMPT.format(text=section.text[:4000])
                }]
            )

            # Parse JSON response
            content = response.content[0].text
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            burdens = []
            for b in data.get("burdens", []):
                burden_type = self._map_to_burden_type(b.get("burden_type", ""))
                burdens.append(IdentifiedBurden(
                    burden_type=burden_type,
                    category=self.taxonomy.categorize_burden(burden_type),
                    severity=self._parse_severity(b.get("severity", "MEDIUM")),
                    location=str(section.citation) if section.citation else section.id,
                    text_excerpt=b.get("text_excerpt", ""),
                    explanation=b.get("estimated_impact", ""),
                    reform_suggestion=b.get("reform_recommendation"),
                    affects_population=b.get("affected_parties"),
                ))

            return burdens

        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return []

    def rewrite_plain_language(self, text: str) -> dict[str, Any]:
        """Rewrite legal text in plain language while preserving meaning"""
        if not self.api_key:
            return {"error": "API key required for plain language rewriting"}

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": self.PLAIN_LANGUAGE_PROMPT.format(text=text[:6000])
            }]
        )

        return {
            "original": text,
            "rewritten": response.content[0].text,
            "mode": "plain_language"
        }

    def analyze_gaps(
        self,
        statute: LegalDocument,
        regulation: LegalDocument,
    ) -> AnalysisResult:
        """Analyze gaps between a statute and its implementing regulation"""
        if not self.api_key:
            return AnalysisResult(
                mode=AnalysisMode.GAP_ANALYSIS,
                document_title=f"{statute.title} vs {regulation.title}",
                findings=[{"error": "API key required for gap analysis"}],
            )

        # Combine section texts
        statute_text = "\n\n".join(s.text for s in statute.sections)[:8000]
        regulation_text = "\n\n".join(s.text for s in regulation.sections)[:8000]

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": self.GAP_ANALYSIS_PROMPT.format(
                    statute=statute_text,
                    regulation=regulation_text
                )
            }]
        )

        content = response.content[0].text
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"raw_response": content}

        return AnalysisResult(
            mode=AnalysisMode.GAP_ANALYSIS,
            document_title=f"{statute.title} vs {regulation.title}",
            findings=data.get("gaps", []),
            summary=data.get("summary", ""),
            confidence_score=data.get("alignment_score", 0) / 100,
        )

    def _get_context(self, text: str, start: int, end: int, context_chars: int = 100) -> str:
        """Get surrounding context for a match"""
        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(text), end + context_chars)
        excerpt = text[ctx_start:ctx_end]
        if ctx_start > 0:
            excerpt = "..." + excerpt
        if ctx_end < len(text):
            excerpt = excerpt + "..."
        return excerpt

    def _map_to_burden_type(self, type_str: str) -> BurdenType:
        """Map LLM-generated burden type string to enum"""
        type_lower = type_str.lower().replace(" ", "_").replace("-", "_")

        for bt in BurdenType:
            if bt.value in type_lower or type_lower in bt.value:
                return bt

        # Default mappings
        if "signature" in type_lower:
            return BurdenType.WET_SIGNATURE
        elif "notar" in type_lower:
            return BurdenType.NOTARIZATION
        elif "person" in type_lower or "appear" in type_lower:
            return BurdenType.IN_PERSON_APPEARANCE
        elif "wait" in type_lower or "period" in type_lower:
            return BurdenType.WAITING_PERIOD
        elif "fee" in type_lower:
            return BurdenType.EXCESSIVE_FEE
        elif "report" in type_lower:
            return BurdenType.MANDATORY_REPORT

        return BurdenType.EXCESSIVE_DOCUMENTATION

    def _parse_severity(self, severity_str: str) -> Severity:
        """Parse severity string to enum"""
        try:
            return Severity[severity_str.upper()]
        except KeyError:
            return Severity.MEDIUM

    def _merge_burden_findings(self, burdens: list[IdentifiedBurden]) -> list[IdentifiedBurden]:
        """Deduplicate and merge burden findings from different sources"""
        seen = {}
        for burden in burdens:
            key = (burden.burden_type, burden.location)
            if key not in seen or burden.reform_suggestion:
                seen[key] = burden
        return list(seen.values())

    def _generate_burden_summary(self, burdens: list[IdentifiedBurden]) -> str:
        """Generate a summary of identified burdens"""
        if not burdens:
            return "No significant procedural burdens identified."

        by_severity = {}
        for b in burdens:
            by_severity.setdefault(b.severity, []).append(b)

        summary_parts = [f"Identified {len(burdens)} procedural burdens:"]

        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            if severity in by_severity:
                count = len(by_severity[severity])
                summary_parts.append(f"- {severity.name}: {count}")

        return "\n".join(summary_parts)

    def _generate_recommendations(self, burdens: list[IdentifiedBurden]) -> list[str]:
        """Generate reform recommendations from identified burdens"""
        recommendations = []

        for burden in burdens:
            if burden.reform_suggestion:
                recommendations.append(burden.reform_suggestion)
            elif burden.severity in [Severity.CRITICAL, Severity.HIGH]:
                recommendations.append(
                    f"Review and simplify {burden.burden_type.value} requirement at {burden.location}"
                )

        return recommendations[:10]  # Top 10 recommendations

    def _burden_to_dict(self, burden: IdentifiedBurden) -> dict[str, Any]:
        """Convert burden to dictionary for JSON serialization"""
        return {
            "burden_type": burden.burden_type.value,
            "category": burden.category.value,
            "severity": burden.severity.name,
            "location": burden.location,
            "text_excerpt": burden.text_excerpt,
            "explanation": burden.explanation,
            "reform_suggestion": burden.reform_suggestion,
            "estimated_time_cost": burden.estimated_time_cost,
            "estimated_dollar_cost": burden.estimated_dollar_cost,
            "affects_population": burden.affects_population,
        }
