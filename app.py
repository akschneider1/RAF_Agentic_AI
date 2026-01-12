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
    initial_sidebar_state="expanded",
)


def main():
    st.title("🏛️ RAF Agentic AI")
    st.subheader("Deproceduralization Tools for Government Reform")

    st.markdown("""
    This prototype demonstrates AI-powered tools for identifying and reforming
    burdensome government procedures, developed in response to the
    [Recoding America Fund](https://www.recodingamerica.fund/)'s Request for Ideas.

    **Powered by the [Agentic State](https://agenticstate.org/) framework vision.**
    """)

    # Sidebar navigation
    st.sidebar.title("Tools")
    tool = st.sidebar.radio(
        "Select a tool:",
        [
            "📋 Burden Scanner",
            "✍️ Plain Language Rewriter",
            "🔍 Gap Analyzer",
            "📊 Cross-State Comparator",
            "📝 Model Bill Drafter",
        ]
    )

    # API key input
    api_key = st.sidebar.text_input(
        "Anthropic API Key (optional)",
        type="password",
        help="For enhanced AI analysis. Pattern-based analysis works without it."
    )

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About

    This prototype addresses the RAF's core requirements:
    - Scan statutory codes for burden sources
    - Identify gaps between statutes and regulations
    - Rewrite regulations in plain language
    - Compare approaches across states
    - Draft model reform legislation
    """)

    # Route to selected tool
    if "Burden Scanner" in tool:
        burden_scanner_ui()
    elif "Plain Language" in tool:
        plain_language_ui()
    elif "Gap Analyzer" in tool:
        gap_analyzer_ui()
    elif "Cross-State" in tool:
        cross_state_ui()
    elif "Model Bill" in tool:
        model_bill_ui()


def burden_scanner_ui():
    """Burden Scanner interface"""
    st.header("📋 Statutory Burden Scanner")

    st.markdown("""
    Scan legal text for procedural burdens including:
    - 📝 Wet signature and notarization requirements
    - 🏢 In-person appearance mandates
    - ⏳ Waiting periods and timeframes
    - 💰 Fee requirements
    - 📑 Reporting obligations
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        sample_text = st.text_area(
            "Enter statutory or regulatory text to scan:",
            height=300,
            placeholder="""Paste legal text here...

Example:
"Section 123. License Application.
(a) Any person seeking a license shall submit an application in writing to the Department.
(b) The application shall be notarized and signed in ink by the applicant.
(c) The applicant shall appear in person before the Board within 30 days.
(d) A non-refundable fee of $500 shall accompany the application..."
""",
            value=SAMPLE_STATUTE,
        )

    with col2:
        st.markdown("### Scan Options")
        use_llm = st.checkbox("Enable AI analysis", value=True,
                              help="Uses Claude for deeper analysis")
        min_severity = st.select_slider(
            "Minimum severity",
            options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            value="LOW"
        )
        jurisdiction = st.text_input("Jurisdiction", value="Sample State")

    if st.button("🔍 Scan for Burdens", type="primary"):
        if not sample_text.strip():
            st.warning("Please enter text to scan.")
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

                # Display results
                st.success(f"Scan complete! Found {len(result.burdens_found)} burdens.")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Burdens", len(result.burdens_found))
                col2.metric("High Priority", len(result.high_priority_findings))
                col3.metric("Sections Analyzed", result.sections_analyzed)
                critical_count = result.summary_stats.get("by_severity", {}).get("CRITICAL", 0)
                col4.metric("Critical Issues", critical_count)

                # Burden breakdown
                st.subheader("Burden Breakdown")
                if result.summary_stats.get("by_type"):
                    burden_data = result.summary_stats["by_type"]
                    st.bar_chart(burden_data)

                # Detailed findings
                st.subheader("Detailed Findings")
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
                        st.markdown(f"**Text:** {burden.text_excerpt}")
                        st.markdown(f"**Category:** {burden.category.value}")
                        if burden.reform_suggestion:
                            st.info(f"💡 **Suggested Reform:** {burden.reform_suggestion}")

                # Recommendations
                if result.recommendations:
                    st.subheader("Recommendations")
                    for rec in result.recommendations:
                        st.markdown(f"• {rec}")

                # Download report
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=result.to_markdown(),
                    file_name="burden_scan_report.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Scan failed: {str(e)}")
                st.exception(e)


def plain_language_ui():
    """Plain Language Rewriter interface"""
    st.header("✍️ Plain Language Rewriter")

    st.markdown("""
    Transform complex legal text into clear, accessible language while
    preserving all legal requirements. Uses federal plain language guidelines.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Text")
        original_text = st.text_area(
            "Enter legal text to simplify:",
            height=300,
            value=SAMPLE_LEGALESE,
        )

        readability_level = st.selectbox(
            "Target audience:",
            ["General Public (8th grade)", "Business Professional", "Technical", "Legal"],
            index=0,
        )

    with col2:
        st.subheader("Plain Language Version")
        rewrite_placeholder = st.empty()

    if st.button("✍️ Rewrite in Plain Language", type="primary"):
        if not original_text.strip():
            st.warning("Please enter text to rewrite.")
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

                # Analyze complexity first
                analysis = rewriter.analyze_complexity(original_text)

                # Perform rewrite
                result = rewriter.rewrite(
                    text=original_text,
                    level=level_map.get(readability_level, ReadabilityLevel.GENERAL_PUBLIC),
                )

                # Display rewritten text
                with col2:
                    rewrite_placeholder.text_area(
                        "Rewritten version:",
                        value=result.rewritten_text,
                        height=300,
                    )

                # Metrics
                st.subheader("Analysis")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Readability Before",
                    f"{result.readability_before:.1f}",
                    help="Flesch Reading Ease (higher = easier)"
                )
                col2.metric(
                    "Readability After",
                    f"{result.readability_after:.1f}",
                    delta=f"+{result.readability_after - result.readability_before:.1f}"
                )
                col3.metric("Legal Confidence", f"{result.confidence_score:.0%}")
                col4.metric("Complexity Rating", analysis["complexity_rating"])

                # Issues found
                st.subheader("Issues Addressed")
                issue_cols = st.columns(3)
                issue_cols[0].metric("Jargon Terms", analysis["jargon_count"])
                issue_cols[1].metric("Passive Voice", analysis["passive_voice_count"])
                issue_cols[2].metric("Long Sentences", len(analysis["long_sentences"]))

                if analysis["jargon_examples"]:
                    st.markdown("**Jargon found:** " + ", ".join(analysis["jargon_examples"]))

                # Warnings
                if result.warnings:
                    st.warning("**Review needed:** " + "; ".join(result.warnings))

                # Changes made
                if result.changes_made:
                    with st.expander("View Changes Made"):
                        for change in result.changes_made:
                            if "before" in change:
                                st.markdown(f"• \"{change['before']}\" → \"{change['after']}\"")
                            elif "description" in change:
                                st.markdown(f"• {change['description']}")

            except Exception as e:
                st.error(f"Rewrite failed: {str(e)}")
                st.exception(e)


def gap_analyzer_ui():
    """Gap Analyzer interface"""
    st.header("🔍 Statute-Regulation Gap Analyzer")

    st.markdown("""
    Compare a regulation against its authorizing statute to identify:
    - Requirements that exceed statutory authority ("gold-plating")
    - Stricter interpretations than required
    - Missing required provisions
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Authorizing Statute")
        statute_text = st.text_area(
            "Enter the statutory text:",
            height=250,
            value=SAMPLE_STATUTE,
            help="The law that authorizes the regulation"
        )

    with col2:
        st.subheader("Implementing Regulation")
        regulation_text = st.text_area(
            "Enter the regulatory text:",
            height=250,
            value=SAMPLE_REGULATION,
            help="The agency's implementing regulation"
        )

    if st.button("🔍 Analyze Gaps", type="primary"):
        if not statute_text.strip() or not regulation_text.strip():
            st.warning("Please enter both statute and regulation text.")
            return

        with st.spinner("Analyzing gaps between statute and regulation..."):
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

                st.success("Gap analysis complete!")

                # Display alignment score
                alignment = result.confidence_score * 100
                if alignment >= 80:
                    st.success(f"📊 Alignment Score: {alignment:.0f}% - Well aligned")
                elif alignment >= 60:
                    st.warning(f"📊 Alignment Score: {alignment:.0f}% - Some gaps found")
                else:
                    st.error(f"📊 Alignment Score: {alignment:.0f}% - Significant gaps")

                # Display findings
                if result.findings:
                    st.subheader("Gaps Identified")
                    for i, gap in enumerate(result.findings, 1):
                        gap_type = gap.get("type", "unknown")
                        icon = "🔴" if "exceeds" in gap_type else "🟡"

                        with st.expander(f"{icon} Gap {i}: {gap_type}"):
                            if gap.get("regulation_text"):
                                st.markdown(f"**Regulation says:** {gap['regulation_text']}")
                            if gap.get("statute_text"):
                                st.markdown(f"**Statute says:** {gap['statute_text']}")
                            if gap.get("analysis"):
                                st.info(gap["analysis"])
                            if gap.get("recommendation"):
                                st.success(f"💡 **Recommendation:** {gap['recommendation']}")

                if result.summary:
                    st.subheader("Summary")
                    st.markdown(result.summary)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)


def cross_state_ui():
    """Cross-State Comparator interface"""
    st.header("📊 Cross-State Comparator")

    st.markdown("""
    Compare regulatory approaches across jurisdictions to identify
    best practices and opportunities for reform.
    """)

    topic = st.selectbox(
        "Regulatory Topic:",
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
        topic = st.text_input("Enter topic:")

    st.subheader("Enter provisions from each jurisdiction")

    num_jurisdictions = st.number_input("Number of jurisdictions to compare:", 2, 5, 2)

    provisions = {}
    cols = st.columns(int(num_jurisdictions))

    for i, col in enumerate(cols):
        with col:
            name = st.text_input(f"Jurisdiction {i+1} name:", value=f"State {chr(65+i)}", key=f"jur_{i}")
            text = st.text_area(f"Provision text:", height=200, key=f"prov_{i}")
            if name and text:
                provisions[name] = text

    if st.button("📊 Compare Jurisdictions", type="primary"):
        if len(provisions) < 2:
            st.warning("Please enter provisions for at least 2 jurisdictions.")
            return

        with st.spinner("Comparing across jurisdictions..."):
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

                st.success("Comparison complete!")

                # Rankings
                st.subheader("Burden Rankings (Least to Most)")
                for i, ranking in enumerate(result.rankings, 1):
                    score = ranking.get("score", "N/A")
                    if isinstance(score, (int, float)):
                        if score <= 3:
                            icon = "🟢"
                        elif score <= 6:
                            icon = "🟡"
                        else:
                            icon = "🔴"
                    else:
                        icon = "⚪"

                    st.markdown(f"{i}. {icon} **{ranking['jurisdiction']}** - Score: {score}")
                    if ranking.get("notes"):
                        st.markdown(f"   _{ranking['notes']}_")

                # Key differences
                if result.key_differences:
                    st.subheader("Key Differences")
                    for diff in result.key_differences:
                        with st.expander(diff.get("aspect", "Difference")):
                            for jur, detail in diff.get("details", {}).items():
                                st.markdown(f"**{jur}:** {detail}")

                # Model provisions
                if result.model_provisions:
                    st.subheader("Model Provisions")
                    for provision in result.model_provisions:
                        with st.expander(f"From {provision.get('source', 'Unknown')}"):
                            st.code(provision.get("text", ""))
                            st.info(f"**Why it works:** {provision.get('rationale', '')}")

                # Recommendations
                if result.recommendations:
                    st.subheader("Recommendations")
                    for rec in result.recommendations:
                        st.markdown(f"• {rec}")

                # Download report
                st.download_button(
                    label="📥 Download Comparison Report",
                    data=result.to_markdown(),
                    file_name="cross_state_comparison.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")
                st.exception(e)


def model_bill_ui():
    """Model Bill Drafter interface"""
    st.header("📝 Model Bill Drafter")

    st.markdown("""
    Generate model legislation based on best practices from other jurisdictions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Law")
        current_law = st.text_area(
            "Enter the current statutory text:",
            height=250,
            value=SAMPLE_STATUTE,
        )
        jurisdiction = st.text_input("Target jurisdiction:", value="State of Example")

    with col2:
        st.subheader("Best Practice Example")
        best_practice = st.text_area(
            "Enter a best practice provision from another jurisdiction:",
            height=250,
            value=SAMPLE_BEST_PRACTICE,
        )

    topic = st.text_input("Topic/Title:", value="License Application Modernization")

    if st.button("📝 Draft Model Bill", type="primary"):
        if not current_law.strip() or not best_practice.strip():
            st.warning("Please enter both current law and best practice text.")
            return

        with st.spinner("Drafting model legislation..."):
            try:
                from src.comparator import StateComparator

                comparator = StateComparator()
                model_bill = comparator.generate_model_bill(
                    topic=topic,
                    current_text=current_law,
                    best_practice_text=best_practice,
                    jurisdiction=jurisdiction,
                )

                st.success("Model bill drafted!")
                st.subheader("Draft Legislation")
                st.text_area(
                    "Model Bill:",
                    value=model_bill,
                    height=400,
                )

                st.download_button(
                    label="📥 Download Model Bill",
                    data=model_bill,
                    file_name="model_bill_draft.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Draft failed: {str(e)}")
                st.exception(e)


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
