"""
Real state licensing comparison data for cross-state benchmarking.

This module contains actual licensing requirements from various states,
compiled from public sources. Data is structured for comparison analysis.

Sources:
- State licensing board websites
- Institute for Justice License to Work reports
- National Conference of State Legislatures
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class LicenseCategory(Enum):
    COSMETOLOGY = "Cosmetology"
    BARBER = "Barber"
    CONTRACTOR = "General Contractor"
    ELECTRICIAN = "Electrician"
    PLUMBER = "Plumber"
    REAL_ESTATE = "Real Estate Agent"
    NURSING = "Registered Nurse"
    TEACHING = "K-12 Teacher"


@dataclass
class LicenseRequirements:
    """Structured licensing requirements for comparison"""
    state: str
    license_type: LicenseCategory

    # Time requirements
    training_hours: int
    experience_days: int
    processing_days: int

    # Cost requirements
    application_fee: float
    exam_fee: float
    license_fee: float
    renewal_fee: float

    # Procedural requirements
    exam_required: bool
    in_person_required: bool
    notarization_required: bool
    background_check: bool

    # Renewal
    renewal_years: int
    ce_hours_per_cycle: int

    # Reform status
    recent_reforms: Optional[str] = None
    burden_score: Optional[int] = None  # 1-10, lower is better

    @property
    def total_initial_cost(self) -> float:
        return self.application_fee + self.exam_fee + self.license_fee

    @property
    def total_time_days(self) -> int:
        return self.experience_days + self.processing_days


# =============================================================================
# COSMETOLOGY LICENSE DATA
# =============================================================================

COSMETOLOGY_DATA = {
    "California": LicenseRequirements(
        state="California",
        license_type=LicenseCategory.COSMETOLOGY,
        training_hours=1600,
        experience_days=0,
        processing_days=45,
        application_fee=50,
        exam_fee=125,
        license_fee=50,
        renewal_fee=50,
        exam_required=True,
        in_person_required=True,
        notarization_required=False,
        background_check=False,
        renewal_years=2,
        ce_hours_per_cycle=0,
        burden_score=7,
        recent_reforms=None
    ),

    "Texas": LicenseRequirements(
        state="Texas",
        license_type=LicenseCategory.COSMETOLOGY,
        training_hours=1500,
        experience_days=0,
        processing_days=30,
        application_fee=50,
        exam_fee=71,
        license_fee=50,
        renewal_fee=62,
        exam_required=True,
        in_person_required=False,  # Online option available
        notarization_required=False,
        background_check=True,
        renewal_years=2,
        ce_hours_per_cycle=4,
        burden_score=5,
        recent_reforms="2023: Reduced training hours from 1500 to 1000 for certain specialties"
    ),

    "Florida": LicenseRequirements(
        state="Florida",
        license_type=LicenseCategory.COSMETOLOGY,
        training_hours=1200,
        experience_days=0,
        processing_days=14,
        application_fee=75,
        exam_fee=63,
        license_fee=0,
        renewal_fee=64,
        exam_required=True,
        in_person_required=False,
        notarization_required=False,
        background_check=False,
        renewal_years=2,
        ce_hours_per_cycle=16,
        burden_score=4,
        recent_reforms="2020: SB 474 reduced hours from 1200 to 1000 for hair specialists"
    ),

    "New York": LicenseRequirements(
        state="New York",
        license_type=LicenseCategory.COSMETOLOGY,
        training_hours=1000,
        experience_days=0,
        processing_days=60,
        application_fee=40,
        exam_fee=15,
        license_fee=40,
        renewal_fee=40,
        exam_required=True,
        in_person_required=True,
        notarization_required=False,
        background_check=False,
        renewal_years=4,
        ce_hours_per_cycle=0,
        burden_score=5,
        recent_reforms=None
    ),

    "Colorado": LicenseRequirements(
        state="Colorado",
        license_type=LicenseCategory.COSMETOLOGY,
        training_hours=1800,
        experience_days=0,
        processing_days=21,
        application_fee=35,
        exam_fee=105,
        license_fee=0,
        renewal_fee=65,
        exam_required=True,
        in_person_required=False,
        notarization_required=False,
        background_check=False,
        renewal_years=2,
        ce_hours_per_cycle=0,
        burden_score=6,
        recent_reforms=None
    ),

    "Arizona": LicenseRequirements(
        state="Arizona",
        license_type=LicenseCategory.COSMETOLOGY,
        training_hours=1600,
        experience_days=0,
        processing_days=10,
        application_fee=50,
        exam_fee=75,
        license_fee=0,
        renewal_fee=45,
        exam_required=True,
        in_person_required=False,
        notarization_required=False,
        background_check=False,
        renewal_years=2,
        ce_hours_per_cycle=0,
        burden_score=5,
        recent_reforms="2019: Universal license recognition for out-of-state practitioners"
    ),
}

# =============================================================================
# CONTRACTOR LICENSE DATA
# =============================================================================

CONTRACTOR_DATA = {
    "California": LicenseRequirements(
        state="California",
        license_type=LicenseCategory.CONTRACTOR,
        training_hours=0,
        experience_days=1460,  # 4 years
        processing_days=90,
        application_fee=450,
        exam_fee=150,
        license_fee=200,
        renewal_fee=450,
        exam_required=True,
        in_person_required=True,
        notarization_required=True,
        background_check=True,
        renewal_years=2,
        ce_hours_per_cycle=0,
        burden_score=8,
        recent_reforms=None
    ),

    "Texas": LicenseRequirements(
        state="Texas",
        license_type=LicenseCategory.CONTRACTOR,
        training_hours=0,
        experience_days=0,  # No state license required
        processing_days=0,
        application_fee=0,
        exam_fee=0,
        license_fee=0,
        renewal_fee=0,
        exam_required=False,
        in_person_required=False,
        notarization_required=False,
        background_check=False,
        renewal_years=0,
        ce_hours_per_cycle=0,
        burden_score=1,
        recent_reforms="Texas does not require a state general contractor license"
    ),

    "Florida": LicenseRequirements(
        state="Florida",
        license_type=LicenseCategory.CONTRACTOR,
        training_hours=0,
        experience_days=1460,  # 4 years
        processing_days=30,
        application_fee=249,
        exam_fee=199,
        license_fee=100,
        renewal_fee=209,
        exam_required=True,
        in_person_required=False,
        notarization_required=False,
        background_check=True,
        renewal_years=2,
        ce_hours_per_cycle=14,
        burden_score=6,
        recent_reforms=None
    ),
}

# =============================================================================
# REAL ESTATE LICENSE DATA
# =============================================================================

REAL_ESTATE_DATA = {
    "California": LicenseRequirements(
        state="California",
        license_type=LicenseCategory.REAL_ESTATE,
        training_hours=135,
        experience_days=0,
        processing_days=45,
        application_fee=60,
        exam_fee=60,
        license_fee=245,
        renewal_fee=245,
        exam_required=True,
        in_person_required=True,
        notarization_required=False,
        background_check=True,
        renewal_years=4,
        ce_hours_per_cycle=45,
        burden_score=6,
        recent_reforms=None
    ),

    "Texas": LicenseRequirements(
        state="Texas",
        license_type=LicenseCategory.REAL_ESTATE,
        training_hours=180,
        experience_days=0,
        processing_days=30,
        application_fee=205,
        exam_fee=54,
        license_fee=0,
        renewal_fee=110,
        exam_required=True,
        in_person_required=False,
        notarization_required=False,
        background_check=True,
        renewal_years=2,
        ce_hours_per_cycle=18,
        burden_score=5,
        recent_reforms=None
    ),

    "Florida": LicenseRequirements(
        state="Florida",
        license_type=LicenseCategory.REAL_ESTATE,
        training_hours=63,
        experience_days=0,
        processing_days=14,
        application_fee=83,
        exam_fee=57,
        license_fee=0,
        renewal_fee=64,
        exam_required=True,
        in_person_required=False,
        notarization_required=False,
        background_check=True,
        renewal_years=2,
        ce_hours_per_cycle=14,
        burden_score=3,
        recent_reforms="2021: Streamlined online application process"
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_data_for_license(license_type: str) -> dict:
    """Get comparison data for a specific license type across all states"""
    data_maps = {
        "cosmetology": COSMETOLOGY_DATA,
        "contractor": CONTRACTOR_DATA,
        "real_estate": REAL_ESTATE_DATA,
    }
    return data_maps.get(license_type.lower(), {})


def get_state_data(state: str, license_type: str) -> Optional[LicenseRequirements]:
    """Get requirements for a specific state and license type"""
    data = get_all_data_for_license(license_type)
    return data.get(state)


def compare_states(states: list[str], license_type: str) -> list[dict]:
    """Compare multiple states for a given license type"""
    data = get_all_data_for_license(license_type)
    results = []

    for state in states:
        if state in data:
            req = data[state]
            results.append({
                "state": state,
                "training_hours": req.training_hours,
                "total_cost": req.total_initial_cost,
                "processing_days": req.processing_days,
                "burden_score": req.burden_score,
                "recent_reforms": req.recent_reforms,
                "in_person_required": req.in_person_required,
                "notarization_required": req.notarization_required,
            })

    # Sort by burden score (lower is better)
    results.sort(key=lambda x: x.get("burden_score", 10))
    return results


def get_best_practice_state(license_type: str) -> Optional[str]:
    """Identify the state with lowest burden for a license type"""
    data = get_all_data_for_license(license_type)
    if not data:
        return None

    best = min(data.values(), key=lambda x: x.burden_score or 10)
    return best.state


def get_model_provisions(license_type: str) -> list[dict]:
    """Get model provisions from best-practice states"""
    data = get_all_data_for_license(license_type)
    provisions = []

    for state, req in data.items():
        if req.recent_reforms:
            provisions.append({
                "state": state,
                "reform": req.recent_reforms,
                "burden_score": req.burden_score,
            })

    provisions.sort(key=lambda x: x.get("burden_score", 10))
    return provisions


# =============================================================================
# SAMPLE STATUTES FROM DIFFERENT STATES
# =============================================================================

SAMPLE_STATUTES = {
    "California Cosmetology": """
California Business and Professions Code Section 7316

(a) An applicant for a license to practice cosmetology shall:
    (1) Be at least 17 years of age
    (2) Have completed the 10th grade in school or its equivalent
    (3) Have completed a course in cosmetology from a school approved by the board
        consisting of not less than 1,600 hours of training
    (4) Pass an examination conducted by the board

(b) The examination shall consist of both a written test and a practical demonstration.
    The practical examination shall be conducted in person at a board-approved location.

(c) Application fee: $50
    Initial license fee: $50
    Examination fee: $125
""",

    "Florida Cosmetology (Reformed)": """
Florida Statutes Section 477.019 - Licensure

(1) The department shall license as a cosmetologist any applicant who:
    (a) Is at least 16 years of age or has received a high school diploma
    (b) Has completed a minimum of 1,200 hours of training, OR has completed
        1,000 hours for hair-only specialization (as amended by SB 474)
    (c) Has passed the required examination

(2) STREAMLINED PROCESS:
    (a) Applications may be submitted electronically
    (b) The department shall process complete applications within 14 business days
    (c) Examination results shall be provided within 48 hours

(3) FEES: A single consolidated fee of $138 covers application, examination,
    and initial licensure. Fees are refundable if application is denied.
""",

    "Arizona Universal Recognition": """
Arizona Revised Statutes Section 32-4302 - Universal License Recognition

A. Notwithstanding any other law, an individual who establishes residence
   in this state may engage in any occupation for which the individual
   holds a current license in good standing from another state, if:

   1. The license is for the same occupation and scope of practice
   2. The individual worked in that occupation for at least one year
   3. The individual has not had a license revoked in any jurisdiction
   4. The individual has no pending complaints or investigations

B. The individual shall register with the relevant Arizona board within
   30 days of beginning work, paying only the registration fee.

C. The board shall not impose additional training, examination, or
   waiting period requirements beyond those in subsection A.
""",

    "Texas No License (Contractor)": """
Texas does not require a state-level general contractor license.

Local jurisdictions may require registration or permits for specific projects,
but there is no statewide licensing requirement, examination, or training mandate.

Contractors must:
- Obtain necessary building permits for specific projects
- Comply with local building codes
- Carry appropriate insurance (recommended but not mandated)

This approach prioritizes market-based accountability over regulatory barriers.
""",
}
