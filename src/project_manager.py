"""
Project management module for RAF Agentic AI.

Provides persistence for reform projects, allowing users to:
- Create and name reform projects
- Save analysis results between sessions
- Track workflow progress through stages
- Export and share projects
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import streamlit as st


class WorkflowStage(Enum):
    """Stages of a reform project workflow"""
    CREATED = "created"
    SCANNED = "scanned"
    ANALYZED = "analyzed"
    DRAFTED = "drafted"
    REVIEWED = "reviewed"
    INTRODUCED = "introduced"
    ENACTED = "enacted"

    @property
    def display_name(self) -> str:
        return self.value.title()

    @property
    def icon(self) -> str:
        icons = {
            "created": "📁",
            "scanned": "🔍",
            "analyzed": "📊",
            "drafted": "📝",
            "reviewed": "✅",
            "introduced": "🏛️",
            "enacted": "⚖️",
        }
        return icons.get(self.value, "📋")

    @classmethod
    def get_stage_order(cls) -> list:
        return [
            cls.CREATED,
            cls.SCANNED,
            cls.ANALYZED,
            cls.DRAFTED,
            cls.REVIEWED,
            cls.INTRODUCED,
            cls.ENACTED,
        ]

    def progress_percentage(self) -> float:
        stages = self.get_stage_order()
        idx = stages.index(self)
        return (idx / (len(stages) - 1)) * 100


@dataclass
class ScanResult:
    """Stored results from a burden scan"""
    timestamp: str
    burdens_found: int
    high_priority: int
    summary: str
    raw_results: Optional[dict] = None


@dataclass
class AnalysisResult:
    """Stored results from gap analysis"""
    timestamp: str
    alignment_score: float
    gaps_found: int
    summary: str
    raw_results: Optional[dict] = None


@dataclass
class DraftResult:
    """Stored draft legislation"""
    timestamp: str
    draft_text: str
    based_on: str
    version: int = 1


@dataclass
class ReformProject:
    """A reform project with all its data and state"""
    # Project identity
    id: str
    name: str
    created_at: str
    updated_at: str

    # Project scope
    state: str
    topic: str
    description: str
    target_statute: str = ""
    target_regulation: str = ""

    # Workflow state
    stage: str = "created"

    # Analysis results
    scan_results: Optional[dict] = None
    analysis_results: Optional[dict] = None
    comparison_results: Optional[dict] = None
    draft_results: Optional[dict] = None

    # Notes and metadata
    notes: list = field(default_factory=list)
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReformProject":
        return cls(**data)

    def get_stage(self) -> WorkflowStage:
        return WorkflowStage(self.stage)

    def advance_stage(self, new_stage: WorkflowStage):
        self.stage = new_stage.value
        self.updated_at = datetime.now().isoformat()

    def add_note(self, note: str):
        self.notes.append({
            "timestamp": datetime.now().isoformat(),
            "text": note,
        })
        self.updated_at = datetime.now().isoformat()


class ProjectManager:
    """Manages reform projects with session-based storage"""

    STORAGE_KEY = "raf_projects"
    CURRENT_PROJECT_KEY = "raf_current_project"

    @classmethod
    def _get_projects_dict(cls) -> dict:
        """Get all projects from session state"""
        if cls.STORAGE_KEY not in st.session_state:
            st.session_state[cls.STORAGE_KEY] = {}
        return st.session_state[cls.STORAGE_KEY]

    @classmethod
    def list_projects(cls) -> list[ReformProject]:
        """List all projects"""
        projects_dict = cls._get_projects_dict()
        return [ReformProject.from_dict(p) for p in projects_dict.values()]

    @classmethod
    def get_project(cls, project_id: str) -> Optional[ReformProject]:
        """Get a specific project by ID"""
        projects_dict = cls._get_projects_dict()
        if project_id in projects_dict:
            return ReformProject.from_dict(projects_dict[project_id])
        return None

    @classmethod
    def save_project(cls, project: ReformProject):
        """Save or update a project"""
        project.updated_at = datetime.now().isoformat()
        projects_dict = cls._get_projects_dict()
        projects_dict[project.id] = project.to_dict()

    @classmethod
    def delete_project(cls, project_id: str):
        """Delete a project"""
        projects_dict = cls._get_projects_dict()
        if project_id in projects_dict:
            del projects_dict[project_id]
        if cls.get_current_project_id() == project_id:
            cls.set_current_project(None)

    @classmethod
    def create_project(
        cls,
        name: str,
        state: str,
        topic: str,
        description: str = "",
    ) -> ReformProject:
        """Create a new project"""
        import uuid
        project = ReformProject(
            id=str(uuid.uuid4())[:8],
            name=name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            state=state,
            topic=topic,
            description=description,
        )
        cls.save_project(project)
        return project

    @classmethod
    def get_current_project_id(cls) -> Optional[str]:
        """Get the currently active project ID"""
        return st.session_state.get(cls.CURRENT_PROJECT_KEY)

    @classmethod
    def get_current_project(cls) -> Optional[ReformProject]:
        """Get the currently active project"""
        project_id = cls.get_current_project_id()
        if project_id:
            return cls.get_project(project_id)
        return None

    @classmethod
    def set_current_project(cls, project_id: Optional[str]):
        """Set the currently active project"""
        st.session_state[cls.CURRENT_PROJECT_KEY] = project_id

    @classmethod
    def export_project(cls, project: ReformProject) -> str:
        """Export project to JSON string"""
        return json.dumps(project.to_dict(), indent=2)

    @classmethod
    def import_project(cls, json_str: str) -> ReformProject:
        """Import project from JSON string"""
        data = json.loads(json_str)
        # Generate new ID to avoid conflicts
        import uuid
        data["id"] = str(uuid.uuid4())[:8]
        data["created_at"] = datetime.now().isoformat()
        data["updated_at"] = datetime.now().isoformat()
        project = ReformProject.from_dict(data)
        cls.save_project(project)
        return project


def render_workflow_progress(project: ReformProject):
    """Render a visual workflow progress indicator"""
    current_stage = project.get_stage()
    stages = WorkflowStage.get_stage_order()

    # Progress bar
    progress = current_stage.progress_percentage() / 100
    st.progress(progress)

    # Stage indicators
    cols = st.columns(len(stages))
    for i, (col, stage) in enumerate(zip(cols, stages)):
        with col:
            is_current = stage == current_stage
            is_complete = stages.index(stage) < stages.index(current_stage)

            if is_complete:
                st.markdown(f"~~{stage.icon}~~")
                st.caption(f"~~{stage.display_name}~~")
            elif is_current:
                st.markdown(f"**{stage.icon}**")
                st.caption(f"**{stage.display_name}**")
            else:
                st.markdown(f"{stage.icon}")
                st.caption(stage.display_name)


def render_project_selector():
    """Render project selector in sidebar"""
    projects = ProjectManager.list_projects()
    current_id = ProjectManager.get_current_project_id()

    if not projects:
        st.info("No projects yet. Create one to get started!")
        return None

    # Build options
    options = {p.id: f"{p.name} ({p.state})" for p in projects}
    options["__none__"] = "— No project selected —"

    # Find current selection index
    if current_id and current_id in options:
        default_idx = list(options.keys()).index(current_id)
    else:
        default_idx = len(options) - 1  # None option

    selected = st.selectbox(
        "Active Project",
        options=list(options.keys()),
        format_func=lambda x: options[x],
        index=default_idx,
        key="project_selector",
    )

    if selected == "__none__":
        ProjectManager.set_current_project(None)
        return None
    else:
        ProjectManager.set_current_project(selected)
        return ProjectManager.get_project(selected)


def render_project_card(project: ReformProject):
    """Render a project summary card"""
    stage = project.get_stage()

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {project.name}")
            st.caption(f"📍 {project.state} • 📋 {project.topic}")
            if project.description:
                st.markdown(project.description[:100] + "..." if len(project.description) > 100 else project.description)

        with col2:
            st.markdown(f"### {stage.icon}")
            st.caption(stage.display_name)

        # Mini progress bar
        st.progress(stage.progress_percentage() / 100)

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Open", key=f"open_{project.id}", use_container_width=True):
                ProjectManager.set_current_project(project.id)
                st.rerun()
        with col2:
            if st.button("Export", key=f"export_{project.id}", use_container_width=True):
                st.download_button(
                    "📥 Download JSON",
                    ProjectManager.export_project(project),
                    f"{project.name.replace(' ', '_')}.json",
                    "application/json",
                    key=f"download_{project.id}",
                )
        with col3:
            if st.button("Delete", key=f"delete_{project.id}", use_container_width=True):
                ProjectManager.delete_project(project.id)
                st.rerun()
