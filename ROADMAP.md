# RAF Agentic AI - Product Roadmap

## Vision

Transform government reform from a manual, fragmented process into an **AI-assisted, continuous workflow** that helps states identify, draft, and track regulatory reforms at scale.

---

## Current State (v0.1 - Prototype)

### What We Have
- **5 standalone tools**: Scanner, Rewriter, Gap Analyzer, Comparator, Bill Drafter
- **Pattern-based analysis**: Works without API key
- **AI-enhanced analysis**: With Anthropic API key
- **Export capabilities**: Download reports as Markdown

### Limitations
- No persistence between sessions
- Manual copy-paste input only
- No real cross-state comparison data
- Tools operate in isolation
- No workflow tracking

---

## Roadmap

### Phase 1: Foundation (Current Sprint)

#### 1.1 Project Persistence
- [x] Save/load reform projects to JSON
- [x] Project metadata: name, state, scope, goals
- [x] Preserve analysis results between sessions
- [x] Project browser in sidebar

#### 1.2 Real Comparison Data
- [x] Pre-loaded licensing laws from 5 states (CA, TX, FL, NY, CO)
- [x] Structured data for meaningful benchmarking
- [x] Actual burden metrics per state

#### 1.3 Workflow State Tracking
- [x] Project stages: Scanned → Analyzed → Drafted → Introduced → Enacted
- [x] Visual progress indicator
- [x] Stage-appropriate tool suggestions

---

### Phase 2: Data Integration (Current)

#### 2.1 Document Upload
- [x] PDF upload and text extraction
- [x] DOCX/DOC support
- [x] TXT file support
- [ ] URL scraping for online legal codes
- [ ] Batch processing multiple documents

#### 2.2 Legal Database Connectors
- [ ] Integration with state legal code APIs
- [ ] Cornell LII integration
- [ ] Automated statute retrieval by citation

#### 2.3 Citation Intelligence
- [ ] Auto-detect statute citations
- [ ] Link findings to specific sections
- [ ] Cross-reference related provisions

---

### Phase 3: Agentic Features

#### 3.1 Proactive Monitoring
- [ ] Agent monitors state registers for new regulations
- [ ] Alerts when new burdens are introduced
- [ ] Weekly digest of regulatory changes

#### 3.2 Auto-Drafting
- [ ] When burden detected, auto-generate reform language
- [ ] Pull from library of proven reform patterns
- [ ] Jurisdiction-specific formatting

#### 3.3 Learning System
- [ ] Track which reforms succeed/fail
- [ ] Improve recommendations based on outcomes
- [ ] Cross-state pattern recognition

---

### Phase 4: Collaboration & Scale

#### 4.1 Multi-User Support
- [ ] User accounts and authentication
- [ ] Role-based access (Analyst, Reviewer, Admin)
- [ ] Team workspaces

#### 4.2 Review Workflows
- [ ] Comment and annotation system
- [ ] Approval workflows for draft legislation
- [ ] Version history and diff views

#### 4.3 Public Dashboard
- [ ] Reform tracker showing bills introduced/passed
- [ ] Impact metrics (time saved, costs reduced)
- [ ] State-by-state comparison leaderboard

---

### Phase 5: Enterprise Features

#### 5.1 API Access
- [ ] RESTful API for bulk analysis
- [ ] Webhook notifications
- [ ] Integration with legislative management systems

#### 5.2 Custom Deployments
- [ ] Self-hosted option for states
- [ ] SSO/SAML integration
- [ ] Audit logging

#### 5.3 Advanced Analytics
- [ ] Fiscal impact estimation
- [ ] Burden reduction forecasting
- [ ] Reform ROI calculator

---

## Architecture Evolution

### Current: Stateless Calculator
```
User → Paste Text → Analyze → Results → (Lost)
```

### Phase 1: Project-Based
```
User → Create Project → Save Analysis → Track Progress → Resume Later
```

### Phase 3+: Agentic Assistant
```
Agent monitors → Detects changes → Alerts user → Suggests reforms → Tracks outcomes
```

---

## Technical Stack

### Current
- **Frontend**: Streamlit
- **AI**: Anthropic Claude API
- **Storage**: Browser session (ephemeral)

### Planned
- **Frontend**: Streamlit → Next.js (Phase 4+)
- **Backend**: FastAPI
- **Database**: PostgreSQL + Vector DB
- **AI**: Claude + embeddings for semantic search
- **Infrastructure**: Cloud-native, containerized

---

## Success Metrics

| Metric | Current | Phase 1 Goal | Phase 3 Goal |
|--------|---------|--------------|--------------|
| States using tool | 0 | 3 pilots | 15 active |
| Reforms tracked | 0 | 50 | 500 |
| Bills drafted | 0 | 10 | 100 |
| Bills enacted | 0 | 1 | 20 |
| Avg. reform cycle time | Unknown | Baseline | -30% |

---

## Contributing

This is an open-source project designed for free adoption by states. Contributions welcome:

1. **Data**: Help us add more state comparison data
2. **Features**: Submit PRs for roadmap items
3. **Testing**: Try the tool and report issues
4. **Advocacy**: Help connect us with state reform teams

---

## Links

- [Recoding America Fund](https://www.recodingamerica.fund/)
- [The Agentic State](https://agenticstate.org/)
- [GitHub Repository](https://github.com/akschneider1/RAF_Agentic_AI)
