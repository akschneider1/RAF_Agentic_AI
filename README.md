---
title: RAF Agentic AI
emoji: 🏛️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
---

# RAF Agentic AI: Deproceduralization Tools Prototype

A prototype suite of AI-powered tools for identifying and reforming burdensome government procedures, developed in response to the Recoding America Fund's Request for Ideas.

## Overview

This project implements diagnostic and reform tools that leverage agentic AI to help states identify and streamline burdensome statutes, regulations, and policies. The architecture is inspired by [The Agentic State](https://agenticstate.org/) framework's vision of proactive, AI-powered government.

## Tools Included

### 1. Statutory Code Scanner
Scans legal text for procedural burdens:
- Wet signature and notary requirements
- In-person appearance mandates
- Waiting periods and timeframes
- Fee requirements
- Reporting obligations

### 2. Plain Language Rewriter
Transforms legalese into clear, accessible language while preserving legal requirements.

### 3. Gap Analyzer
Compares regulations against authorizing statutes to identify "gold-plating" where agencies exceed statutory authority.

### 4. Cross-State Comparator
Benchmarks regulatory approaches across jurisdictions to identify best practices.

### 5. Model Bill Drafter
Generates draft reform legislation based on best-practice examples.

## Usage

1. Select a tool from the sidebar
2. Paste or enter legal text
3. Click the action button to analyze
4. Review findings and recommendations

**For enhanced AI analysis:** Enter your Anthropic API key in the sidebar. Pattern-based analysis works without it.

## Links

- [Recoding America Fund](https://www.recodingamerica.fund/)
- [The Agentic State](https://agenticstate.org/)
- [GitHub Repository](https://github.com/akschneider1/RAF_Agentic_AI)

## License

Open source under MIT License - designed for free adoption by states per RAF's mission.
