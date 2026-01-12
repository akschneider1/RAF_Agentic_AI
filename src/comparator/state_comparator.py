"""
Cross-State Comparator - Benchmark regulations across jurisdictions

Addresses RAF RFI Requirement #5a:
"Conducting cross-state analysis that compares a state's statutes and regulations
to those of other states and potentially assesses their relative burden"
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..core.burden_taxonomy import BurdenType, Severity
from ..core.document import DocumentType, LegalDocument


class ComparisonDimension(Enum):
    """Dimensions for comparing regulations"""
    FEES = "fees"
    TIMEFRAMES = "timeframes"
    DOCUMENTATION = "documentation"
    APPROVALS = "approvals"
    ACCESSIBILITY = "accessibility"
    OVERALL_BURDEN = "overall_burden"


@dataclass
class JurisdictionProfile:
    """Profile of a jurisdiction's regulatory approach"""
    name: str
    documents: list[LegalDocument] = field(default_factory=list)
    burden_scores: dict[str, float] = field(default_factory=dict)
    notable_provisions: list[dict[str, Any]] = field(default_factory=list)
    best_practices: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of cross-jurisdiction comparison"""
    topic: str
    jurisdictions: list[str]
    dimension: ComparisonDimension
    rankings: list[dict[str, Any]]  # Ranked from least to most burdensome
    key_differences: list[dict[str, Any]]
    recommendations: list[str]
    model_provisions: list[dict[str, Any]]

    def to_markdown(self) -> str:
        """Generate markdown comparison report"""
        lines = [
            f"# Cross-State Comparison: {self.topic}",
            f"*Comparison Dimension: {self.dimension.value}*",
            "",
            "## Jurisdiction Rankings (Least to Most Burdensome)",
            "",
        ]

        for i, ranking in enumerate(self.rankings, 1):
            lines.append(
                f"{i}. **{ranking['jurisdiction']}** - Score: {ranking.get('score', 'N/A')}"
            )
            if ranking.get('notes'):
                lines.append(f"   - {ranking['notes']}")

        lines.extend([
            "",
            "## Key Differences",
            "",
        ])

        for diff in self.key_differences:
            lines.append(f"### {diff.get('aspect', 'General')}")
            for jurisdiction, detail in diff.get('details', {}).items():
                lines.append(f"- **{jurisdiction}**: {detail}")
            lines.append("")

        if self.model_provisions:
            lines.extend([
                "## Model Provisions (Best Practices)",
                "",
            ])
            for provision in self.model_provisions:
                lines.append(f"### From {provision.get('source', 'Unknown')}")
                lines.append(f"```")
                lines.append(provision.get('text', ''))
                lines.append(f"```")
                lines.append(f"*Why it works:* {provision.get('rationale', '')}")
                lines.append("")

        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)


class StateComparator:
    """
    Compare regulatory approaches across states/jurisdictions.

    This tool helps identify:
    - Which states have the least burdensome approaches
    - Best practices that could be adopted
    - Specific provisions that could serve as models
    - Opportunities for harmonization
    """

    COMPARISON_PROMPT = """You are analyzing regulatory provisions across multiple jurisdictions
to identify differences in burden and best practices.

Compare the following provisions on the topic of "{topic}":

{provisions}

For each jurisdiction, assess:
1. Overall burden level (1-10, where 10 is most burdensome)
2. Specific requirements (fees, timeframes, documentation)
3. Notable features (good or bad)

Then provide:
1. Rankings from least to most burdensome with explanations
2. Key differences between jurisdictions
3. Model provisions that represent best practices
4. Recommendations for the target jurisdiction

Format as JSON:
{{
    "rankings": [
        {{"jurisdiction": "...", "score": 1-10, "notes": "..."}}
    ],
    "key_differences": [
        {{"aspect": "...", "details": {{"State A": "...", "State B": "..."}}}}
    ],
    "model_provisions": [
        {{"source": "...", "text": "...", "rationale": "..."}}
    ],
    "recommendations": ["..."]
}}"""

    # Common regulatory topics for comparison
    STANDARD_TOPICS = [
        "business_licensing",
        "professional_licensing",
        "building_permits",
        "environmental_permits",
        "vehicle_registration",
        "food_service_permits",
        "healthcare_licensing",
        "contractor_licensing",
        "alcohol_licensing",
        "childcare_licensing",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        self.profiles: dict[str, JurisdictionProfile] = {}

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

    def add_jurisdiction(
        self,
        name: str,
        documents: list[LegalDocument],
    ) -> JurisdictionProfile:
        """Add a jurisdiction's documents for comparison"""
        profile = JurisdictionProfile(name=name, documents=documents)
        self.profiles[name] = profile
        return profile

    def compare(
        self,
        topic: str,
        jurisdictions: list[str],
        provisions: dict[str, str],  # jurisdiction -> provision text
        dimension: ComparisonDimension = ComparisonDimension.OVERALL_BURDEN,
    ) -> ComparisonResult:
        """
        Compare provisions across jurisdictions.

        Args:
            topic: The regulatory topic (e.g., "business_licensing")
            jurisdictions: List of jurisdiction names to compare
            provisions: Dict mapping jurisdiction name to provision text
            dimension: Primary comparison dimension

        Returns:
            ComparisonResult with rankings and recommendations
        """
        if not self.api_key:
            return self._basic_comparison(topic, jurisdictions, provisions, dimension)

        # Format provisions for LLM
        formatted_provisions = "\n\n".join([
            f"=== {jurisdiction} ===\n{text}"
            for jurisdiction, text in provisions.items()
        ])

        try:
            import json

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": self.COMPARISON_PROMPT.format(
                        topic=topic,
                        provisions=formatted_provisions[:10000]
                    )
                }]
            )

            content = response.content[0].text

            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            return ComparisonResult(
                topic=topic,
                jurisdictions=jurisdictions,
                dimension=dimension,
                rankings=data.get("rankings", []),
                key_differences=data.get("key_differences", []),
                recommendations=data.get("recommendations", []),
                model_provisions=data.get("model_provisions", []),
            )

        except Exception as e:
            print(f"LLM comparison failed: {e}")
            return self._basic_comparison(topic, jurisdictions, provisions, dimension)

    def _basic_comparison(
        self,
        topic: str,
        jurisdictions: list[str],
        provisions: dict[str, str],
        dimension: ComparisonDimension,
    ) -> ComparisonResult:
        """Basic pattern-based comparison without LLM"""
        import re

        rankings = []

        for jurisdiction, text in provisions.items():
            score = self._calculate_burden_score(text)
            rankings.append({
                "jurisdiction": jurisdiction,
                "score": score,
                "notes": f"Based on {len(text.split())} words of regulation",
            })

        # Sort by score (lower is better)
        rankings.sort(key=lambda x: x["score"])

        # Identify differences
        key_differences = self._identify_differences(provisions)

        return ComparisonResult(
            topic=topic,
            jurisdictions=jurisdictions,
            dimension=dimension,
            rankings=rankings,
            key_differences=key_differences,
            recommendations=self._generate_basic_recommendations(rankings),
            model_provisions=[],
        )

    def _calculate_burden_score(self, text: str) -> float:
        """Calculate a basic burden score from text"""
        import re

        score = 5.0  # Start at middle

        # Factors that increase burden
        burden_indicators = [
            (r"\b(shall|must|required)\b", 0.1),
            (r"\bnotari", 0.5),
            (r"\bin person\b", 0.5),
            (r"\bwithin \d+ days\b", 0.2),
            (r"\bfee\b", 0.2),
            (r"\bpublic hearing\b", 0.3),
            (r"\bwritten (notice|consent)\b", 0.2),
        ]

        for pattern, weight in burden_indicators:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * weight

        # Factors that reduce burden
        ease_indicators = [
            (r"\bonline\b", -0.3),
            (r"\belectronic(ally)?\b", -0.3),
            (r"\bmay\b", -0.1),
            (r"\bwaiv(e|er)\b", -0.2),
        ]

        for pattern, weight in ease_indicators:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * weight

        # Normalize to 1-10
        return max(1.0, min(10.0, score))

    def _identify_differences(
        self,
        provisions: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Identify key differences between provisions"""
        import re

        differences = []

        # Check for fee differences
        fee_info = {}
        for jurisdiction, text in provisions.items():
            fees = re.findall(r"\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", text)
            if fees:
                fee_info[jurisdiction] = ", ".join(f"${f}" for f in fees[:3])
            else:
                fee_info[jurisdiction] = "Not specified"

        if len(set(fee_info.values())) > 1:
            differences.append({
                "aspect": "Fees",
                "details": fee_info,
            })

        # Check for timeframe differences
        time_info = {}
        for jurisdiction, text in provisions.items():
            times = re.findall(r"(\d+)\s*(day|week|month|year)s?", text, re.IGNORECASE)
            if times:
                time_info[jurisdiction] = ", ".join(f"{n} {u}s" for n, u in times[:3])
            else:
                time_info[jurisdiction] = "Not specified"

        if len(set(time_info.values())) > 1:
            differences.append({
                "aspect": "Timeframes",
                "details": time_info,
            })

        # Check for digital options
        digital_info = {}
        for jurisdiction, text in provisions.items():
            has_online = bool(re.search(r"\b(online|electronic|digital)\b", text, re.IGNORECASE))
            digital_info[jurisdiction] = "Available" if has_online else "Not mentioned"

        if len(set(digital_info.values())) > 1:
            differences.append({
                "aspect": "Digital/Online Options",
                "details": digital_info,
            })

        return differences

    def _generate_basic_recommendations(
        self,
        rankings: list[dict[str, Any]],
    ) -> list[str]:
        """Generate basic recommendations from rankings"""
        recommendations = []

        if len(rankings) >= 2:
            best = rankings[0]["jurisdiction"]
            worst = rankings[-1]["jurisdiction"]

            recommendations.append(
                f"Consider adopting approaches from {best}, which has the lowest burden score."
            )

            if rankings[-1]["score"] > 7:
                recommendations.append(
                    f"Significant simplification opportunities exist - {worst} has a high burden score of {rankings[-1]['score']:.1f}."
                )

        recommendations.append(
            "Conduct detailed analysis of specific provisions to identify model language."
        )

        return recommendations

    def generate_model_bill(
        self,
        topic: str,
        current_text: str,
        best_practice_text: str,
        jurisdiction: str,
    ) -> str:
        """
        Generate model bill language based on best practices.

        Addresses RAF RFI Requirement #5b:
        "Drafting model bill language to streamline burdensome statutes"
        """
        if not self.api_key:
            return "API key required for model bill generation."

        prompt = f"""Draft model bill language for {jurisdiction} to reform the following provision.

CURRENT {jurisdiction.upper()} LAW:
{current_text}

BEST PRACTICE EXAMPLE:
{best_practice_text}

Create model bill language that:
1. Follows {jurisdiction}'s legislative drafting style
2. Incorporates best practices from the example
3. Reduces procedural burden while maintaining necessary protections
4. Includes appropriate effective date and transition provisions

Format as:
SECTION 1. [Title of Section]
[Bill text...]

SECTION 2. EFFECTIVE DATE
[Effective date provision...]

EXPLANATORY NOTES:
[Brief explanation of changes and their purpose]"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text
