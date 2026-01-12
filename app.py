"""
RAF Agentic AI - Streamlit Demo Interface

Interactive demonstration of deproceduralization tools for the
Recoding America Fund's state reform initiatives.
"""

import os
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="RAF Agentic AI - Deproceduralization Tools",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better UX
st.markdown("""
<style>
    /* Card-like containers */
    .tool-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .tool-card:hover {
        transform: translateY(-2px);
    }

    /* Better button styling */
    .stButton > button[kind="primary"] {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
    }

    /* Cleaner tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 500;
    }

    /* Info boxes */
    .info-box {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Hide default hamburger menu on mobile for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "current_tool" not in st.session_state:
        st.session_state.current_tool = "home"
    if "show_samples" not in st.session_state:
        st.session_state.show_samples = False


def render_sidebar():
    """Render a clean, minimal sidebar"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # API key with better UX
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Required for AI-powered analysis. Get one at console.anthropic.com",
            placeholder="sk-ant-..."
        )

        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            st.success("✓ API key configured", icon="✅")
        else:
            st.info("Pattern-based analysis available without API key", icon="ℹ️")

        st.divider()

        # Quick links
        st.markdown("### 🔗 Resources")
        st.markdown("""
        - [Recoding America Fund](https://www.recodingamerica.fund/)
        - [The Agentic State](https://agenticstate.org/)
        - [GitHub Repo](https://github.com/akschneider1/RAF_Agentic_AI)
        """)

        st.divider()

        # Sample text toggle
        st.session_state.show_samples = st.toggle(
            "Pre-fill sample text",
            value=st.session_state.show_samples,
            help="Populate text areas with example content"
        )


def render_home():
    """Render the home/landing page with tool cards"""

    # Hero section
    st.markdown("""
    # 🏛️ RAF Agentic AI
    ### AI-Powered Tools for Government Reform

    Identify and reform burdensome government procedures with our suite of diagnostic tools.
    Developed for the [Recoding America Fund](https://www.recodingamerica.fund/)'s mission to modernize state government.
    """)

    st.divider()

    # Tool cards in a grid
    st.markdown("## Choose a Tool")
    st.caption("Select a tool below to get started, or use the tabs above to navigate.")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 📋 Burden Scanner")
            st.markdown("Scan statutory and regulatory text to identify procedural burdens like signature requirements, waiting periods, and fees.")
            if st.button("Launch Scanner →", key="home_scanner", use_container_width=True):
                st.session_state.current_tool = "scanner"
                st.rerun()

        with st.container(border=True):
            st.markdown("### 🔍 Gap Analyzer")
            st.markdown("Compare regulations against authorizing statutes to find 'gold-plating' where agencies exceed their authority.")
            if st.button("Launch Analyzer →", key="home_gap", use_container_width=True):
                st.session_state.current_tool = "gap"
                st.rerun()

        with st.container(border=True):
            st.markdown("### 📝 Model Bill Drafter")
            st.markdown("Generate reform legislation based on best practices from other jurisdictions.")
            if st.button("Launch Drafter →", key="home_bill", use_container_width=True):
                st.session_state.current_tool = "bill"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### ✍️ Plain Language Rewriter")
            st.markdown("Transform complex legal jargon into clear, accessible language while preserving legal meaning.")
            if st.button("Launch Rewriter →", key="home_rewriter", use_container_width=True):
                st.session_state.current_tool = "rewriter"
                st.rerun()

        with st.container(border=True):
            st.markdown("### 📊 Cross-State Comparator")
            st.markdown("Benchmark regulatory approaches across states to identify best practices and reform opportunities.")
            if st.button("Launch Comparator →", key="home_compare", use_container_width=True):
                st.session_state.current_tool = "compare"
                st.rerun()

        # Getting started tips
        with st.container(border=True):
            st.markdown("### 💡 Getting Started")
            st.markdown("""
            1. **New to the tools?** Start with the **Burden Scanner** to analyze existing law
            2. **Have an API key?** Add it in Settings (sidebar) for AI-powered analysis
            3. **Just exploring?** Toggle "Pre-fill sample text" in the sidebar
            """)


def render_tool_header(icon: str, title: str, description: str):
    """Render a consistent tool header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"# {icon} {title}")
        st.caption(description)
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.current_tool = "home"
            st.rerun()
    st.divider()


def burden_scanner_ui():
    """Burden Scanner interface"""
    render_tool_header(
        "📋",
        "Burden Scanner",
        "Identify procedural burdens in statutory and regulatory text"
    )

    # Quick guide
    with st.expander("ℹ️ What does this tool detect?", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Documentation**\n- 📝 Wet signatures\n- 🔏 Notarization\n- 📄 Paper forms")
        with col2:
            st.markdown("**Access Barriers**\n- 🏢 In-person visits\n- ⏳ Waiting periods\n- 💰 Fees")
        with col3:
            st.markdown("**Compliance**\n- 📑 Reporting rules\n- 🔄 Renewal requirements\n- 📋 Documentation")

    # Main input area
    sample_text = st.text_area(
        "**Paste your statutory or regulatory text:**",
        height=250,
        placeholder="Paste legal text here to scan for procedural burdens...",
        value=SAMPLE_STATUTE if st.session_state.show_samples else "",
    )

    # Options in a cleaner layout
    with st.expander("⚙️ Scan Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            use_llm = st.checkbox(
                "Enable AI analysis",
                value=True,
                help="Uses Claude for deeper analysis (requires API key)"
            )
        with col2:
            min_severity = st.select_slider(
                "Minimum severity",
                options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                value="LOW"
            )
        with col3:
            jurisdiction = st.text_input("Jurisdiction", value="Sample State")

    # Primary action button
    if st.button("🔍 Scan for Burdens", type="primary", use_container_width=True):
        if not sample_text.strip():
            st.warning("⚠️ Please enter text to scan.")
            return

        with st.spinner("Scanning for procedural burdens..."):
            try:
                from src.scanner import StatuteScanner
                from src.scanner.statute_scanner import ScanConfig
                from src.core.burden_taxonomy import Severity

                config = ScanConfig(
                    use_llm=use_llm and bool(os.getenv("ANTHROPIC_API_KEY")),
                    min_severity=Severity[min_severity],
                )

                scanner = StatuteScanner()
                result = scanner.scan_text(
                    text=sample_text,
                    title="User Input",
                    jurisdiction=jurisdiction,
                    config=config,
                )

                # Results section
                st.success(f"✅ Scan complete! Found **{len(result.burdens_found)} burdens**.")

                # Summary metrics in cards
                st.markdown("### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Burdens", len(result.burdens_found))
                col2.metric("High Priority", len(result.high_priority_findings))
                col3.metric("Sections Analyzed", result.sections_analyzed)
                critical_count = result.summary_stats.get("by_severity", {}).get("CRITICAL", 0)
                col4.metric("Critical Issues", critical_count)

                # Burden breakdown chart
                if result.summary_stats.get("by_type"):
                    st.markdown("### 📈 Burden Breakdown")
                    st.bar_chart(result.summary_stats["by_type"])

                # Detailed findings
                if result.burdens_found:
                    st.markdown("### 🔎 Detailed Findings")
                    for i, burden in enumerate(result.burdens_found, 1):
                        severity_colors = {
                            "CRITICAL": "🔴",
                            "HIGH": "🟠",
                            "MEDIUM": "🟡",
                            "LOW": "🟢",
                        }
                        icon = severity_colors.get(burden.severity.name, "⚪")

                        with st.expander(f"{icon} {burden.burden_type.value} ({burden.severity.name})"):
                            st.markdown(f"**Location:** {burden.location}")
                            st.markdown(f"**Text:** _{burden.text_excerpt}_")
                            st.markdown(f"**Category:** {burden.category.value}")
                            if burden.reform_suggestion:
                                st.info(f"💡 **Reform Suggestion:** {burden.reform_suggestion}")

                # Recommendations
                if result.recommendations:
                    st.markdown("### 💡 Recommendations")
                    for rec in result.recommendations:
                        st.markdown(f"- {rec}")

                # Download
                st.divider()
                st.download_button(
                    label="📥 Download Full Report",
                    data=result.to_markdown(),
                    file_name="burden_scan_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"❌ Scan failed: {str(e)}")
                with st.expander("Error details"):
                    st.exception(e)


def plain_language_ui():
    """Plain Language Rewriter interface"""
    render_tool_header(
        "✍️",
        "Plain Language Rewriter",
        "Transform complex legal text into clear, accessible language"
    )

    # Side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Original Text")
        original_text = st.text_area(
            "Paste complex legal text:",
            height=300,
            placeholder="Paste legalese here...",
            value=SAMPLE_LEGALESE if st.session_state.show_samples else "",
            label_visibility="collapsed",
        )

        readability_level = st.selectbox(
            "Target audience:",
            ["General Public (8th grade)", "Business Professional", "Technical", "Legal"],
            index=0,
        )

    with col2:
        st.markdown("### ✨ Plain Language Version")
        result_placeholder = st.empty()
        result_placeholder.text_area(
            "Result:",
            height=300,
            placeholder="Rewritten text will appear here...",
            disabled=True,
            label_visibility="collapsed",
        )

    if st.button("✍️ Rewrite in Plain Language", type="primary", use_container_width=True):
        if not original_text.strip():
            st.warning("⚠️ Please enter text to rewrite.")
            return

        with st.spinner("Rewriting in plain language..."):
            try:
                from src.rewriter import PlainLanguageRewriter
                from src.rewriter.plain_language import ReadabilityLevel

                level_map = {
                    "General Public (8th grade)": ReadabilityLevel.GENERAL_PUBLIC,
                    "Business Professional": ReadabilityLevel.BUSINESS,
                    "Technical": ReadabilityLevel.TECHNICAL,
                    "Legal": ReadabilityLevel.LEGAL,
                }

                rewriter = PlainLanguageRewriter()
                analysis = rewriter.analyze_complexity(original_text)
                result = rewriter.rewrite(
                    text=original_text,
                    level=level_map.get(readability_level, ReadabilityLevel.GENERAL_PUBLIC),
                )

                # Update result area
                with col2:
                    result_placeholder.text_area(
                        "Result:",
                        value=result.rewritten_text,
                        height=300,
                        label_visibility="collapsed",
                    )

                # Metrics
                st.markdown("### 📊 Readability Analysis")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Before",
                    f"{result.readability_before:.0f}",
                    help="Flesch Reading Ease"
                )
                col2.metric(
                    "After",
                    f"{result.readability_after:.0f}",
                    delta=f"+{result.readability_after - result.readability_before:.0f}"
                )
                col3.metric("Confidence", f"{result.confidence_score:.0%}")
                col4.metric("Complexity", analysis["complexity_rating"])

                # Issues addressed
                st.markdown("### 🔧 Issues Addressed")
                issue_cols = st.columns(3)
                issue_cols[0].metric("Jargon Terms", analysis["jargon_count"])
                issue_cols[1].metric("Passive Voice", analysis["passive_voice_count"])
                issue_cols[2].metric("Long Sentences", len(analysis["long_sentences"]))

                if analysis["jargon_examples"]:
                    st.caption("**Jargon found:** " + ", ".join(analysis["jargon_examples"]))

                if result.warnings:
                    st.warning("⚠️ **Review needed:** " + "; ".join(result.warnings))

            except Exception as e:
                st.error(f"❌ Rewrite failed: {str(e)}")
                with st.expander("Error details"):
                    st.exception(e)


def gap_analyzer_ui():
    """Gap Analyzer interface"""
    render_tool_header(
        "🔍",
        "Gap Analyzer",
        "Compare regulations against authorizing statutes to identify overreach"
    )

    with st.expander("ℹ️ What does this tool find?", expanded=False):
        st.markdown("""
        - **Gold-plating**: Requirements that exceed statutory authority
        - **Stricter interpretations**: Tighter rules than the law requires
        - **Missing provisions**: Required elements not implemented
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📜 Authorizing Statute")
        statute_text = st.text_area(
            "The law that authorizes the regulation:",
            height=250,
            placeholder="Paste the statutory text here...",
            value=SAMPLE_STATUTE if st.session_state.show_samples else "",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("### 📋 Implementing Regulation")
        regulation_text = st.text_area(
            "The agency's implementing regulation:",
            height=250,
            placeholder="Paste the regulatory text here...",
            value=SAMPLE_REGULATION if st.session_state.show_samples else "",
            label_visibility="collapsed",
        )

    if st.button("🔍 Analyze Gaps", type="primary", use_container_width=True):
        if not statute_text.strip() or not regulation_text.strip():
            st.warning("⚠️ Please enter both statute and regulation text.")
            return

        with st.spinner("Analyzing gaps..."):
            try:
                from src.core.agent import AgenticEngine
                from src.core.document import DocumentProcessor, DocumentType

                processor = DocumentProcessor()
                statute = processor.parse_text(
                    text=statute_text,
                    doc_type=DocumentType.STATUTE,
                    title="Authorizing Statute",
                )
                regulation = processor.parse_text(
                    text=regulation_text,
                    doc_type=DocumentType.REGULATION,
                    title="Implementing Regulation",
                )

                engine = AgenticEngine()
                result = engine.analyze_gaps(statute, regulation)

                # Alignment score with visual indicator
                alignment = result.confidence_score * 100
                st.markdown("### 📊 Alignment Score")

                if alignment >= 80:
                    st.success(f"**{alignment:.0f}%** — Well aligned with statutory authority")
                elif alignment >= 60:
                    st.warning(f"**{alignment:.0f}%** — Some gaps identified")
                else:
                    st.error(f"**{alignment:.0f}%** — Significant gaps found")

                st.progress(alignment / 100)

                # Findings
                if result.findings:
                    st.markdown("### 🔎 Gaps Identified")
                    for i, gap in enumerate(result.findings, 1):
                        gap_type = gap.get("type", "unknown")
                        icon = "🔴" if "exceeds" in gap_type.lower() else "🟡"

                        with st.expander(f"{icon} Gap {i}: {gap_type}"):
                            if gap.get("regulation_text"):
                                st.markdown(f"**Regulation:** _{gap['regulation_text']}_")
                            if gap.get("statute_text"):
                                st.markdown(f"**Statute:** _{gap['statute_text']}_")
                            if gap.get("analysis"):
                                st.info(gap["analysis"])
                            if gap.get("recommendation"):
                                st.success(f"💡 **Recommendation:** {gap['recommendation']}")

                if result.summary:
                    st.markdown("### 📝 Summary")
                    st.markdown(result.summary)

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                with st.expander("Error details"):
                    st.exception(e)


def cross_state_ui():
    """Cross-State Comparator interface"""
    render_tool_header(
        "📊",
        "Cross-State Comparator",
        "Compare regulatory approaches across jurisdictions"
    )

    topic = st.selectbox(
        "**Regulatory Topic:**",
        [
            "Business Licensing",
            "Professional Licensing",
            "Building Permits",
            "Environmental Permits",
            "Food Service Permits",
            "Contractor Licensing",
            "Custom Topic..."
        ]
    )

    if topic == "Custom Topic...":
        topic = st.text_input("Enter custom topic:")

    st.markdown("### Enter Provisions by Jurisdiction")

    num_jurisdictions = st.slider("Number of jurisdictions:", 2, 5, 2)

    provisions = {}
    cols = st.columns(int(num_jurisdictions))

    for i, col in enumerate(cols):
        with col:
            name = st.text_input(
                f"State {i+1}:",
                value=f"State {chr(65+i)}",
                key=f"jur_{i}"
            )
            text = st.text_area(
                "Provision:",
                height=200,
                key=f"prov_{i}",
                placeholder="Paste regulatory text...",
            )
            if name and text:
                provisions[name] = text

    if st.button("📊 Compare Jurisdictions", type="primary", use_container_width=True):
        if len(provisions) < 2:
            st.warning("⚠️ Please enter provisions for at least 2 jurisdictions.")
            return

        with st.spinner("Comparing jurisdictions..."):
            try:
                from src.comparator import StateComparator
                from src.comparator.state_comparator import ComparisonDimension

                comparator = StateComparator()
                result = comparator.compare(
                    topic=topic,
                    jurisdictions=list(provisions.keys()),
                    provisions=provisions,
                    dimension=ComparisonDimension.OVERALL_BURDEN,
                )

                st.success("✅ Comparison complete!")

                # Rankings
                st.markdown("### 🏆 Burden Rankings (Least to Most)")
                for i, ranking in enumerate(result.rankings, 1):
                    score = ranking.get("score", "N/A")
                    if isinstance(score, (int, float)):
                        icon = "🟢" if score <= 3 else "🟡" if score <= 6 else "🔴"
                    else:
                        icon = "⚪"

                    st.markdown(f"**{i}. {icon} {ranking['jurisdiction']}** — Score: {score}")
                    if ranking.get("notes"):
                        st.caption(f"_{ranking['notes']}_")

                # Key differences
                if result.key_differences:
                    st.markdown("### 🔀 Key Differences")
                    for diff in result.key_differences:
                        with st.expander(diff.get("aspect", "Difference")):
                            for jur, detail in diff.get("details", {}).items():
                                st.markdown(f"**{jur}:** {detail}")

                # Model provisions
                if result.model_provisions:
                    st.markdown("### ⭐ Model Provisions")
                    for provision in result.model_provisions:
                        with st.expander(f"From {provision.get('source', 'Unknown')}"):
                            st.code(provision.get("text", ""))
                            st.info(f"**Why it works:** {provision.get('rationale', '')}")

                # Recommendations
                if result.recommendations:
                    st.markdown("### 💡 Recommendations")
                    for rec in result.recommendations:
                        st.markdown(f"- {rec}")

                st.divider()
                st.download_button(
                    label="📥 Download Report",
                    data=result.to_markdown(),
                    file_name="cross_state_comparison.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"❌ Comparison failed: {str(e)}")
                with st.expander("Error details"):
                    st.exception(e)


def model_bill_ui():
    """Model Bill Drafter interface"""
    render_tool_header(
        "📝",
        "Model Bill Drafter",
        "Generate reform legislation from best practice examples"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📜 Current Law")
        current_law = st.text_area(
            "Current statutory text:",
            height=250,
            placeholder="Paste the current law you want to reform...",
            value=SAMPLE_STATUTE if st.session_state.show_samples else "",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("### ⭐ Best Practice Example")
        best_practice = st.text_area(
            "Best practice from another jurisdiction:",
            height=250,
            placeholder="Paste a model provision to emulate...",
            value=SAMPLE_BEST_PRACTICE if st.session_state.show_samples else "",
            label_visibility="collapsed",
        )

    col1, col2 = st.columns(2)
    with col1:
        jurisdiction = st.text_input("Target jurisdiction:", value="State of Example")
    with col2:
        topic = st.text_input("Bill title:", value="License Application Modernization Act")

    if st.button("📝 Draft Model Bill", type="primary", use_container_width=True):
        if not current_law.strip() or not best_practice.strip():
            st.warning("⚠️ Please enter both current law and best practice text.")
            return

        with st.spinner("Drafting legislation..."):
            try:
                from src.comparator import StateComparator

                comparator = StateComparator()
                model_bill = comparator.generate_model_bill(
                    topic=topic,
                    current_text=current_law,
                    best_practice_text=best_practice,
                    jurisdiction=jurisdiction,
                )

                st.success("✅ Model bill drafted!")

                st.markdown("### 📄 Draft Legislation")
                st.text_area(
                    "Model Bill:",
                    value=model_bill,
                    height=400,
                    label_visibility="collapsed",
                )

                st.download_button(
                    label="📥 Download Model Bill",
                    data=model_bill,
                    file_name="model_bill_draft.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"❌ Draft failed: {str(e)}")
                with st.expander("Error details"):
                    st.exception(e)


def main():
    """Main application entry point"""
    init_session_state()
    render_sidebar()

    # Tab-based navigation
    tab_home, tab_scanner, tab_rewriter, tab_gap, tab_compare, tab_bill = st.tabs([
        "🏠 Home",
        "📋 Scanner",
        "✍️ Rewriter",
        "🔍 Gap Analyzer",
        "📊 Comparator",
        "📝 Bill Drafter",
    ])

    with tab_home:
        render_home()

    with tab_scanner:
        burden_scanner_ui()

    with tab_rewriter:
        plain_language_ui()

    with tab_gap:
        gap_analyzer_ui()

    with tab_compare:
        cross_state_ui()

    with tab_bill:
        model_bill_ui()


# Sample texts for demonstration
SAMPLE_STATUTE = """Section 123. License Application Requirements.

(a) Application Form. Any person seeking a professional license pursuant to this chapter shall submit an application to the Department in writing on forms prescribed by the Director. The application shall be signed in ink by the applicant.

(b) Notarization. The application shall be notarized by a notary public commissioned in this state.

(c) Personal Appearance. Prior to the issuance of any license, the applicant shall appear in person before the Board at a regularly scheduled meeting.

(d) Waiting Period. No license shall be issued until at least 45 days after the submission of a complete application.

(e) Fees. The following non-refundable fees shall accompany each application:
    (1) Application fee: $500
    (2) Background check fee: $150
    (3) Processing fee: $75

(f) Annual Report. Each licensee shall submit an annual report to the Department, in writing, detailing all professional activities conducted during the preceding calendar year.

(g) Renewal. Licenses shall expire on December 31 of each year. Applications for renewal must be submitted by certified mail not less than 60 days prior to expiration."""

SAMPLE_REGULATION = """Administrative Code Section 123.01 - Implementation of License Application Requirements

(a) In addition to the requirements set forth in Section 123 of the Code, applicants shall provide:
    (1) Three letters of recommendation, each notarized;
    (2) Certified copies of all educational transcripts;
    (3) A personal statement of not less than 500 words;
    (4) Proof of liability insurance in the amount of $1,000,000.

(b) The application fee shall be $750 to account for administrative costs.

(c) Applicants must appear in person at the Department office for fingerprinting prior to their Board appearance. The fingerprinting appointment must be scheduled at least 30 days in advance.

(d) Following Board approval, applicants shall wait an additional 15 business days before the license is issued to allow for final processing.

(e) The annual report required by Section 123(f) shall include:
    (1) A detailed log of all client interactions;
    (2) Copies of any complaints received;
    (3) Proof of continuing education completion;
    (4) Updated liability insurance documentation."""

SAMPLE_LEGALESE = """Notwithstanding any provision of this Agreement to the contrary, and pursuant to the terms and conditions set forth herein, the Party of the First Part (hereinafter referred to as "Licensor") shall, upon receipt of the requisite documentation as specified in Schedule A attached hereto and incorporated herein by reference, be obligated to issue, within a timeframe not to exceed forty-five (45) business days from the date of submission of a complete application, the appropriate licensure documentation to the Party of the Second Part (hereinafter referred to as "Licensee"), provided that the Licensee shall have satisfied all prerequisites, including but not limited to the submission of notarized affidavits, the payment of all applicable fees, and the personal appearance before the duly constituted Board at such time and place as shall be determined by the Director in his or her sole and absolute discretion."""

SAMPLE_BEST_PRACTICE = """Section 45. Streamlined License Application.

(a) Online Application. Applications for professional licensure may be submitted electronically through the Department's secure online portal. Electronic signatures are accepted.

(b) Document Upload. Required documentation may be uploaded in electronic format. The Department shall accept unofficial transcripts for initial review.

(c) Virtual Options. Applicants may elect to appear before the Board via secure video conference.

(d) Processing Time. The Department shall process complete applications within 15 business days. Applicants shall receive real-time status updates through the online portal.

(e) Single Fee. A consolidated application fee of $200 shall cover all processing costs. The fee is refundable if the application is denied.

(f) Automatic Renewal. Licenses shall renew automatically upon payment of the renewal fee and confirmation of continuing education completion. No paper renewal application is required."""


if __name__ == "__main__":
    main()
