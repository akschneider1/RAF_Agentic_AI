"""
Burden Taxonomy - Classification system for procedural burdens

Based on research from the Recoding America Fund's deproceduralization framework
and administrative burden literature.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BurdenCategory(Enum):
    """Primary categories of procedural burden"""

    # Learning costs - effort to understand requirements
    LEARNING = "learning"

    # Compliance costs - effort to fulfill requirements
    COMPLIANCE = "compliance"

    # Psychological costs - stress, stigma, loss of autonomy
    PSYCHOLOGICAL = "psychological"

    # Administrative costs - paperwork, waiting, redundancy
    ADMINISTRATIVE = "administrative"

    # Opportunity costs - time/resources that could be used elsewhere
    OPPORTUNITY = "opportunity"


class BurdenType(Enum):
    """Specific types of procedural burdens identified in statutes/regulations"""

    # Document requirements
    WET_SIGNATURE = "wet_signature"
    NOTARIZATION = "notarization"
    CERTIFIED_COPY = "certified_copy"
    ORIGINAL_DOCUMENT = "original_document"

    # Reporting requirements
    MANDATORY_REPORT = "mandatory_report"
    PERIODIC_FILING = "periodic_filing"
    DUPLICATE_REPORTING = "duplicate_reporting"

    # Meeting/appearance requirements
    IN_PERSON_APPEARANCE = "in_person_appearance"
    MANDATORY_MEETING = "mandatory_meeting"
    PUBLIC_HEARING = "public_hearing"

    # Approval requirements
    MULTI_AGENCY_APPROVAL = "multi_agency_approval"
    SEQUENTIAL_APPROVAL = "sequential_approval"
    REDUNDANT_APPROVAL = "redundant_approval"

    # Time-based burdens
    WAITING_PERIOD = "waiting_period"
    EXPIRATION_REQUIREMENT = "expiration_requirement"
    RENEWAL_REQUIREMENT = "renewal_requirement"

    # Information collection
    EXCESSIVE_DOCUMENTATION = "excessive_documentation"
    REDUNDANT_INFORMATION = "redundant_information"
    OUTDATED_FORM = "outdated_form"

    # Fee-based burdens
    EXCESSIVE_FEE = "excessive_fee"
    MULTIPLE_FEES = "multiple_fees"
    NON_REFUNDABLE_FEE = "non_refundable_fee"

    # Structural issues
    CONFLICTING_REQUIREMENT = "conflicting_requirement"
    DUPLICATIVE_STATUTE = "duplicative_statute"
    OUTDATED_REFERENCE = "outdated_reference"
    VAGUE_STANDARD = "vague_standard"

    # Process issues
    NO_DIGITAL_OPTION = "no_digital_option"
    PAPER_ONLY = "paper_only"
    MAIL_ONLY = "mail_only"


class Severity(Enum):
    """Severity levels for identified burdens"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BurdenIndicator:
    """Patterns that indicate presence of a burden type"""
    burden_type: BurdenType
    patterns: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    context_clues: list[str] = field(default_factory=list)
    typical_severity: Severity = Severity.MEDIUM


@dataclass
class IdentifiedBurden:
    """A specific burden identified in a document"""
    burden_type: BurdenType
    category: BurdenCategory
    severity: Severity
    location: str  # Citation or location in document
    text_excerpt: str
    explanation: str
    reform_suggestion: Optional[str] = None
    estimated_time_cost: Optional[str] = None
    estimated_dollar_cost: Optional[str] = None
    affects_population: Optional[str] = None


class BurdenTaxonomy:
    """
    Comprehensive taxonomy for classifying procedural burdens.

    This taxonomy is designed to help identify and categorize the types
    of procedural requirements that create unnecessary friction in
    government services.
    """

    def __init__(self):
        self.indicators = self._build_indicators()

    def _build_indicators(self) -> dict[BurdenType, BurdenIndicator]:
        """Build the pattern indicators for each burden type"""

        return {
            BurdenType.WET_SIGNATURE: BurdenIndicator(
                burden_type=BurdenType.WET_SIGNATURE,
                patterns=[
                    r"original signature",
                    r"signed in ink",
                    r"hand-?signed",
                    r"wet signature",
                    r"physical signature",
                ],
                keywords=["signature", "sign", "signed", "signing"],
                context_clues=["original", "ink", "physical", "handwritten"],
                typical_severity=Severity.MEDIUM
            ),

            BurdenType.NOTARIZATION: BurdenIndicator(
                burden_type=BurdenType.NOTARIZATION,
                patterns=[
                    r"notari[zs]ed",
                    r"notary public",
                    r"acknowledged before",
                    r"sworn before",
                ],
                keywords=["notary", "notarized", "notarization", "acknowledged"],
                context_clues=["public", "seal", "commission"],
                typical_severity=Severity.HIGH
            ),

            BurdenType.IN_PERSON_APPEARANCE: BurdenIndicator(
                burden_type=BurdenType.IN_PERSON_APPEARANCE,
                patterns=[
                    r"appear in person",
                    r"personal appearance",
                    r"physically present",
                    r"appear before",
                    r"in-?person",
                ],
                keywords=["appear", "appearance", "present", "person"],
                context_clues=["office", "court", "agency", "department"],
                typical_severity=Severity.HIGH
            ),

            BurdenType.MANDATORY_REPORT: BurdenIndicator(
                burden_type=BurdenType.MANDATORY_REPORT,
                patterns=[
                    r"shall (submit|file|provide) a report",
                    r"required to report",
                    r"annual report",
                    r"quarterly report",
                    r"monthly report",
                ],
                keywords=["report", "reporting", "submit", "file"],
                context_clues=["annual", "quarterly", "monthly", "periodic"],
                typical_severity=Severity.MEDIUM
            ),

            BurdenType.WAITING_PERIOD: BurdenIndicator(
                burden_type=BurdenType.WAITING_PERIOD,
                patterns=[
                    r"\d+\s*days? (before|after|prior|following)",
                    r"waiting period",
                    r"cooling[- ]off period",
                    r"not (less|fewer) than \d+",
                ],
                keywords=["wait", "waiting", "period", "days"],
                context_clues=["before", "after", "prior", "minimum"],
                typical_severity=Severity.MEDIUM
            ),

            BurdenType.MULTI_AGENCY_APPROVAL: BurdenIndicator(
                burden_type=BurdenType.MULTI_AGENCY_APPROVAL,
                patterns=[
                    r"approval (of|from|by) .+ (and|,) .+ (department|agency|board)",
                    r"coordinate with",
                    r"consultation with .+ (agency|department)",
                ],
                keywords=["approval", "coordinate", "consultation", "agencies"],
                context_clues=["multiple", "both", "all", "each"],
                typical_severity=Severity.HIGH
            ),

            BurdenType.CONFLICTING_REQUIREMENT: BurdenIndicator(
                burden_type=BurdenType.CONFLICTING_REQUIREMENT,
                patterns=[
                    r"notwithstanding .+ (section|provision)",
                    r"except as provided",
                    r"in conflict with",
                ],
                keywords=["notwithstanding", "except", "conflict", "contrary"],
                context_clues=["provision", "section", "statute", "regulation"],
                typical_severity=Severity.CRITICAL
            ),

            BurdenType.EXCESSIVE_FEE: BurdenIndicator(
                burden_type=BurdenType.EXCESSIVE_FEE,
                patterns=[
                    r"fee of \$?\d+",
                    r"filing fee",
                    r"application fee",
                    r"processing fee",
                ],
                keywords=["fee", "fees", "cost", "charge", "payment"],
                context_clues=["filing", "application", "processing", "license"],
                typical_severity=Severity.MEDIUM  # Requires comparison analysis
            ),

            BurdenType.PAPER_ONLY: BurdenIndicator(
                burden_type=BurdenType.PAPER_ONLY,
                patterns=[
                    r"in writing",
                    r"written (notice|application|request)",
                    r"mail(ed)? to",
                    r"by (certified|registered) mail",
                ],
                keywords=["writing", "written", "mail", "paper", "hard copy"],
                context_clues=["submit", "send", "deliver", "provide"],
                typical_severity=Severity.MEDIUM
            ),

            BurdenType.VAGUE_STANDARD: BurdenIndicator(
                burden_type=BurdenType.VAGUE_STANDARD,
                patterns=[
                    r"as (the|an?) .+ (deems|determines) (appropriate|necessary)",
                    r"reasonable",
                    r"adequate",
                    r"sufficient",
                    r"good cause",
                ],
                keywords=["reasonable", "adequate", "appropriate", "sufficient"],
                context_clues=["determine", "discretion", "judgment", "deem"],
                typical_severity=Severity.LOW
            ),
        }

    def get_indicator(self, burden_type: BurdenType) -> Optional[BurdenIndicator]:
        """Get the indicator patterns for a burden type"""
        return self.indicators.get(burden_type)

    def get_all_patterns(self) -> dict[BurdenType, list[str]]:
        """Get all regex patterns organized by burden type"""
        return {
            bt: indicator.patterns
            for bt, indicator in self.indicators.items()
        }

    def categorize_burden(self, burden_type: BurdenType) -> BurdenCategory:
        """Map a burden type to its primary category"""

        category_mapping = {
            # Administrative burdens
            BurdenType.WET_SIGNATURE: BurdenCategory.ADMINISTRATIVE,
            BurdenType.NOTARIZATION: BurdenCategory.ADMINISTRATIVE,
            BurdenType.CERTIFIED_COPY: BurdenCategory.ADMINISTRATIVE,
            BurdenType.PAPER_ONLY: BurdenCategory.ADMINISTRATIVE,
            BurdenType.MAIL_ONLY: BurdenCategory.ADMINISTRATIVE,
            BurdenType.EXCESSIVE_DOCUMENTATION: BurdenCategory.ADMINISTRATIVE,
            BurdenType.REDUNDANT_INFORMATION: BurdenCategory.ADMINISTRATIVE,

            # Compliance burdens
            BurdenType.MANDATORY_REPORT: BurdenCategory.COMPLIANCE,
            BurdenType.PERIODIC_FILING: BurdenCategory.COMPLIANCE,
            BurdenType.DUPLICATE_REPORTING: BurdenCategory.COMPLIANCE,
            BurdenType.MULTI_AGENCY_APPROVAL: BurdenCategory.COMPLIANCE,

            # Opportunity costs
            BurdenType.WAITING_PERIOD: BurdenCategory.OPPORTUNITY,
            BurdenType.IN_PERSON_APPEARANCE: BurdenCategory.OPPORTUNITY,
            BurdenType.MANDATORY_MEETING: BurdenCategory.OPPORTUNITY,

            # Learning costs
            BurdenType.VAGUE_STANDARD: BurdenCategory.LEARNING,
            BurdenType.CONFLICTING_REQUIREMENT: BurdenCategory.LEARNING,

            # Psychological costs (often overlap with financial)
            BurdenType.EXCESSIVE_FEE: BurdenCategory.PSYCHOLOGICAL,
            BurdenType.NON_REFUNDABLE_FEE: BurdenCategory.PSYCHOLOGICAL,
        }

        return category_mapping.get(burden_type, BurdenCategory.ADMINISTRATIVE)
