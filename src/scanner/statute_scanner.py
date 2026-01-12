"""
Statute Scanner - Diagnostic tool for scanning statutory codes

Addresses RAF RFI Requirement #1:
"Diagnostic tools that can easily scan a state's statutory codes for sources of burden"
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from ..core.agent import AgenticEngine, AnalysisResult
from ..core.burden_taxonomy import BurdenTaxonomy, BurdenType, IdentifiedBurden, Severity
from ..core.document import DocumentProcessor, DocumentType, LegalDocument, Section


@dataclass
class ScanConfig:
    """Configuration for statutory scan"""
    # Which burden types to scan for
    burden_types: list[BurdenType] = field(default_factory=lambda: list(BurdenType))

    # Minimum severity to report
    min_severity: Severity = Severity.LOW

    # Whether to use LLM for deeper analysis
    use_llm: bool = True

    # Maximum sections to analyze with LLM (to manage costs)
    max_llm_sections: int = 50

    # Output format
    output_format: str = "json"  # json, markdown, html


@dataclass
class ScanResult:
    """Complete result of a statutory code scan"""
    jurisdiction: str
    documents_scanned: int
    sections_analyzed: int
    burdens_found: list[IdentifiedBurden]
    summary_stats: dict[str, Any]
    high_priority_findings: list[IdentifiedBurden]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "documents_scanned": self.documents_scanned,
            "sections_analyzed": self.sections_analyzed,
            "total_burdens": len(self.burdens_found),
            "summary_stats": self.summary_stats,
            "high_priority_count": len(self.high_priority_findings),
            "burdens": [
                {
                    "type": b.burden_type.value,
                    "category": b.category.value,
                    "severity": b.severity.name,
                    "location": b.location,
                    "excerpt": b.text_excerpt[:200],
                    "reform": b.reform_suggestion,
                }
                for b in self.burdens_found
            ],
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """Generate markdown report"""
        lines = [
            f"# Statutory Burden Scan: {self.jurisdiction}",
            "",
            "## Summary",
            f"- Documents scanned: {self.documents_scanned}",
            f"- Sections analyzed: {self.sections_analyzed}",
            f"- Total burdens identified: {len(self.burdens_found)}",
            f"- High priority findings: {len(self.high_priority_findings)}",
            "",
            "## Burden Distribution",
        ]

        for category, count in self.summary_stats.get("by_category", {}).items():
            lines.append(f"- {category}: {count}")

        lines.extend([
            "",
            "## High Priority Findings",
            "",
        ])

        for i, burden in enumerate(self.high_priority_findings[:10], 1):
            lines.extend([
                f"### {i}. {burden.burden_type.value} ({burden.severity.name})",
                f"**Location:** {burden.location}",
                f"**Text:** {burden.text_excerpt[:300]}...",
                f"**Reform Suggestion:** {burden.reform_suggestion or 'Review and simplify'}",
                "",
            ])

        lines.extend([
            "## Recommendations",
            "",
        ])

        for i, rec in enumerate(self.recommendations[:10], 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)


class StatuteScanner:
    """
    Diagnostic tool for scanning statutory codes for procedural burdens.

    This is the primary tool for the first phase of deproceduralization work,
    helping states identify specific statutes that create unnecessary burden.

    Usage:
        scanner = StatuteScanner()
        result = scanner.scan_directory("./state_statutes/")
        print(result.to_markdown())
    """

    # Specific patterns for statutory language
    STATUTORY_BURDEN_PATTERNS = {
        "wet_signature": [
            r"shall (?:be )?sign(?:ed)? (?:by|in)",
            r"signature (?:of|by) the",
            r"personally sign",
            r"original signature",
        ],
        "notarization": [
            r"shall be notarized",
            r"acknowledged before a notary",
            r"notary public",
            r"notarial (?:seal|certificate)",
        ],
        "in_person": [
            r"shall appear (?:in person|before)",
            r"personal appearance",
            r"physically present",
            r"appear at (?:a|the) (?:office|hearing|meeting)",
        ],
        "mandatory_report": [
            r"shall (?:submit|file|provide) (?:a |an )?(?:annual|quarterly|monthly)? ?report",
            r"report(?:ing)? requirement",
            r"shall report to (?:the )?(?:department|agency|board)",
        ],
        "waiting_period": [
            r"(?:at least |not less than |no fewer than )\d+ (?:day|week|month)",
            r"waiting period of \d+",
            r"\d+ day(?:s)? (?:prior to|before|after)",
            r"cooling[- ]off period",
        ],
        "publication": [
            r"shall (?:be )?publish(?:ed)? (?:in|for)",
            r"notice shall be published",
            r"publication in (?:a |the )?newspaper",
        ],
        "hearing": [
            r"public hearing shall be held",
            r"opportunity for (?:a )?hearing",
            r"hearing before (?:the|a)",
        ],
        "approval_chain": [
            r"approval (?:of|from|by) (?:the )?(?:director|commissioner|secretary|board|department)",
            r"with the consent of",
            r"subject to (?:the )?approval",
        ],
        "excessive_fee": [
            r"fee (?:of|not (?:to )?exceed(?:ing)?) \$?\d+",
            r"filing fee",
            r"application fee",
            r"license fee of",
        ],
        "paper_requirement": [
            r"in writing",
            r"written (?:notice|application|request|consent)",
            r"by (?:certified|registered) mail",
            r"delivered by mail",
        ],
        "outdated_reference": [
            r"as (?:it )?existed on (?:January|February|March|April|May|June|July|August|September|October|November|December) \d+, (?:19|20)\d{2}",
            r"prior to (?:January|February|March|April|May|June|July|August|September|October|November|December) \d+, (?:19|20)\d{2}",
        ],
        "conflicting": [
            r"notwithstanding (?:any )?(?:other )?(?:provision|section|law)",
            r"except as (?:otherwise )?provided",
            r"in the event of (?:a )?conflict",
        ],
    }

    def __init__(self, api_key: Optional[str] = None):
        self.processor = DocumentProcessor()
        self.engine = AgenticEngine(api_key=api_key)
        self.taxonomy = BurdenTaxonomy()

    def scan_text(
        self,
        text: str,
        title: str = "Untitled Statute",
        jurisdiction: str = "Unknown",
        config: Optional[ScanConfig] = None,
    ) -> ScanResult:
        """Scan a single text for burdens"""
        config = config or ScanConfig()

        document = self.processor.parse_text(
            text=text,
            doc_type=DocumentType.STATUTE,
            title=title,
            jurisdiction=jurisdiction,
        )

        return self._scan_document(document, config)

    def scan_file(
        self,
        filepath: Path,
        jurisdiction: str = "Unknown",
        config: Optional[ScanConfig] = None,
    ) -> ScanResult:
        """Scan a single file for burdens"""
        config = config or ScanConfig()
        document = self.processor.parse_file(filepath)
        document.jurisdiction = jurisdiction

        return self._scan_document(document, config)

    def scan_directory(
        self,
        directory: Path,
        jurisdiction: str = "Unknown",
        config: Optional[ScanConfig] = None,
        file_pattern: str = "*.txt",
    ) -> ScanResult:
        """Scan all matching files in a directory"""
        config = config or ScanConfig()
        directory = Path(directory)

        all_burdens = []
        documents_scanned = 0
        sections_analyzed = 0

        for filepath in directory.glob(file_pattern):
            try:
                document = self.processor.parse_file(filepath)
                document.jurisdiction = jurisdiction

                result = self._scan_document(document, config)
                all_burdens.extend(result.burdens_found)
                documents_scanned += 1
                sections_analyzed += result.sections_analyzed
            except Exception as e:
                print(f"Error scanning {filepath}: {e}")

        return self._compile_results(
            jurisdiction=jurisdiction,
            documents_scanned=documents_scanned,
            sections_analyzed=sections_analyzed,
            burdens=all_burdens,
        )

    def _scan_document(
        self,
        document: LegalDocument,
        config: ScanConfig,
    ) -> ScanResult:
        """Scan a single document"""
        all_burdens = []
        llm_analyzed = 0

        for section in document.iter_sections():
            # Pattern-based scan (always)
            pattern_burdens = self._pattern_scan(section)
            all_burdens.extend(pattern_burdens)

            # LLM analysis (if enabled and under limit)
            if config.use_llm and llm_analyzed < config.max_llm_sections:
                if section.word_count > 30:  # Only analyze substantial sections
                    result = self.engine.analyze_for_burdens(
                        LegalDocument(
                            title=document.title,
                            doc_type=document.doc_type,
                            source=document.source,
                            sections=[section],
                        ),
                        use_llm=True,
                    )
                    # Convert findings back to IdentifiedBurden
                    for finding in result.findings:
                        burden = self._finding_to_burden(finding, section)
                        if burden:
                            all_burdens.append(burden)
                    llm_analyzed += 1

        # Filter by configured severity
        filtered_burdens = [
            b for b in all_burdens
            if b.severity.value >= config.min_severity.value
        ]

        return self._compile_results(
            jurisdiction=document.jurisdiction or "Unknown",
            documents_scanned=1,
            sections_analyzed=document.total_sections,
            burdens=filtered_burdens,
        )

    def _pattern_scan(self, section: Section) -> list[IdentifiedBurden]:
        """Fast pattern-based burden detection"""
        burdens = []
        text = section.text

        for burden_name, patterns in self.STATUTORY_BURDEN_PATTERNS.items():
            burden_type = self._name_to_burden_type(burden_name)

            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    excerpt = self._extract_context(text, match.start(), match.end())

                    burdens.append(IdentifiedBurden(
                        burden_type=burden_type,
                        category=self.taxonomy.categorize_burden(burden_type),
                        severity=self._estimate_severity(burden_type, excerpt),
                        location=str(section.citation) if section.citation else section.id,
                        text_excerpt=excerpt,
                        explanation=f"Statutory {burden_name.replace('_', ' ')} requirement",
                    ))

        return burdens

    def _name_to_burden_type(self, name: str) -> BurdenType:
        """Map pattern name to BurdenType"""
        mapping = {
            "wet_signature": BurdenType.WET_SIGNATURE,
            "notarization": BurdenType.NOTARIZATION,
            "in_person": BurdenType.IN_PERSON_APPEARANCE,
            "mandatory_report": BurdenType.MANDATORY_REPORT,
            "waiting_period": BurdenType.WAITING_PERIOD,
            "publication": BurdenType.MANDATORY_REPORT,
            "hearing": BurdenType.PUBLIC_HEARING,
            "approval_chain": BurdenType.MULTI_AGENCY_APPROVAL,
            "excessive_fee": BurdenType.EXCESSIVE_FEE,
            "paper_requirement": BurdenType.PAPER_ONLY,
            "outdated_reference": BurdenType.OUTDATED_REFERENCE,
            "conflicting": BurdenType.CONFLICTING_REQUIREMENT,
        }
        return mapping.get(name, BurdenType.EXCESSIVE_DOCUMENTATION)

    def _estimate_severity(self, burden_type: BurdenType, excerpt: str) -> Severity:
        """Estimate severity based on burden type and context"""
        # Base severity from taxonomy
        indicator = self.taxonomy.get_indicator(burden_type)
        base_severity = indicator.typical_severity if indicator else Severity.MEDIUM

        # Escalate based on context clues
        escalation_triggers = [
            ("shall", 0),
            ("must", 0),
            ("required", 0),
            ("mandatory", 1),
            ("penalty", 1),
            ("violation", 1),
            ("felony", 2),
            ("void", 1),
            ("invalid", 1),
        ]

        escalation = 0
        excerpt_lower = excerpt.lower()
        for trigger, value in escalation_triggers:
            if trigger in excerpt_lower:
                escalation = max(escalation, value)

        # Apply escalation
        new_value = min(base_severity.value + escalation, Severity.CRITICAL.value)
        return Severity(new_value)

    def _extract_context(self, text: str, start: int, end: int, context: int = 150) -> str:
        """Extract context around a match"""
        ctx_start = max(0, start - context)
        ctx_end = min(len(text), end + context)

        excerpt = text[ctx_start:ctx_end].strip()
        if ctx_start > 0:
            excerpt = "..." + excerpt
        if ctx_end < len(text):
            excerpt = excerpt + "..."

        return excerpt

    def _finding_to_burden(self, finding: dict, section: Section) -> Optional[IdentifiedBurden]:
        """Convert engine finding dict to IdentifiedBurden"""
        try:
            burden_type = BurdenType(finding.get("burden_type", "excessive_documentation"))
            return IdentifiedBurden(
                burden_type=burden_type,
                category=self.taxonomy.categorize_burden(burden_type),
                severity=Severity[finding.get("severity", "MEDIUM")],
                location=finding.get("location", section.id),
                text_excerpt=finding.get("text_excerpt", ""),
                explanation=finding.get("explanation", ""),
                reform_suggestion=finding.get("reform_suggestion"),
                affects_population=finding.get("affects_population"),
            )
        except (ValueError, KeyError):
            return None

    def _compile_results(
        self,
        jurisdiction: str,
        documents_scanned: int,
        sections_analyzed: int,
        burdens: list[IdentifiedBurden],
    ) -> ScanResult:
        """Compile scan results into summary"""

        # Compute stats
        by_type = {}
        by_category = {}
        by_severity = {}

        for burden in burdens:
            by_type[burden.burden_type.value] = by_type.get(burden.burden_type.value, 0) + 1
            by_category[burden.category.value] = by_category.get(burden.category.value, 0) + 1
            by_severity[burden.severity.name] = by_severity.get(burden.severity.name, 0) + 1

        summary_stats = {
            "by_type": by_type,
            "by_category": by_category,
            "by_severity": by_severity,
        }

        # High priority = CRITICAL or HIGH severity
        high_priority = [
            b for b in burdens
            if b.severity in [Severity.CRITICAL, Severity.HIGH]
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(burdens)

        return ScanResult(
            jurisdiction=jurisdiction,
            documents_scanned=documents_scanned,
            sections_analyzed=sections_analyzed,
            burdens_found=burdens,
            summary_stats=summary_stats,
            high_priority_findings=high_priority,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, burdens: list[IdentifiedBurden]) -> list[str]:
        """Generate actionable recommendations from findings"""
        recommendations = []

        # Group by type
        by_type = {}
        for b in burdens:
            by_type.setdefault(b.burden_type, []).append(b)

        # Top recommendations by frequency and severity
        type_priority = sorted(
            by_type.items(),
            key=lambda x: (
                max(b.severity.value for b in x[1]),
                len(x[1])
            ),
            reverse=True
        )

        for burden_type, instances in type_priority[:5]:
            count = len(instances)
            if burden_type == BurdenType.WET_SIGNATURE:
                recommendations.append(
                    f"Enable e-signatures: Found {count} wet signature requirements. "
                    "Adopt UETA/E-SIGN compliant electronic signature standards."
                )
            elif burden_type == BurdenType.NOTARIZATION:
                recommendations.append(
                    f"Allow remote notarization: Found {count} notarization requirements. "
                    "Adopt remote online notarization (RON) to reduce in-person burden."
                )
            elif burden_type == BurdenType.IN_PERSON_APPEARANCE:
                recommendations.append(
                    f"Enable virtual options: Found {count} in-person requirements. "
                    "Allow video conferencing alternatives where feasible."
                )
            elif burden_type == BurdenType.WAITING_PERIOD:
                recommendations.append(
                    f"Review waiting periods: Found {count} mandatory delays. "
                    "Assess whether each waiting period serves a legitimate purpose."
                )
            elif burden_type == BurdenType.PAPER_ONLY:
                recommendations.append(
                    f"Digitize paper processes: Found {count} paper-only requirements. "
                    "Enable online submission and digital document acceptance."
                )
            else:
                recommendations.append(
                    f"Review {burden_type.value}: Found {count} instances. "
                    "Assess necessity and potential for streamlining."
                )

        return recommendations


# CLI interface
def main():
    """Command-line interface for statute scanner"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan statutory codes for procedural burdens"
    )
    parser.add_argument("input", help="File or directory to scan")
    parser.add_argument("--jurisdiction", "-j", default="Unknown", help="Jurisdiction name")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--format", "-f", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM analysis")

    args = parser.parse_args()

    config = ScanConfig(use_llm=not args.no_llm)
    scanner = StatuteScanner()

    input_path = Path(args.input)
    if input_path.is_dir():
        result = scanner.scan_directory(input_path, args.jurisdiction, config)
    else:
        result = scanner.scan_file(input_path, args.jurisdiction, config)

    if args.format == "json":
        output = json.dumps(result.to_dict(), indent=2)
    else:
        output = result.to_markdown()

    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
