import os
from pathlib import Path
from typing import Any, Dict, Optional
import json

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from src.utils.file_parser import FileParser, FileParsingError
from src.config.nova_config import NovaConfig
from src.config import ConfigLoader
from src.nova_graph import NovaGraph


APP_TITLE = "Nova HR — Professional CV Analyzer"
RESULTS_DIR = Path("results")


def init_state() -> None:
    """Initialize session state variables"""
    if "nova" not in st.session_state:
        config = ConfigLoader.load_nova_config()
        st.session_state.nova = NovaGraph(config)
    st.session_state.setdefault("position_description", "")
    st.session_state.setdefault("uploaded_file_name", None)
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("progress_step", 0)
    st.session_state.setdefault("theme", "Midnight")


def set_page_config() -> None:
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    """Inject custom CSS for professional styling"""
    st.markdown("""
        <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main container styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(102, 126, 234, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 300;
            margin: 0;
            letter-spacing: -0.5px;
        }
        
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.95;
            margin-top: 0.5rem;
        }
        
        /* Score badge styling */
        .score-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
            text-align: center;
        }
        
        .score-excellent {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
        }
        
        .score-good {
            background: linear-gradient(135deg, #17a2b8 0%, #5bc0de 100%);
            color: white;
        }
        
        .score-fair {
            background: linear-gradient(135deg, #ffc107 0%, #ffdb4d 100%);
            color: #333;
        }
        
        .score-poor {
            background: linear-gradient(135deg, #dc3545 0%, #ff6b7a 100%);
            color: white;
        }
        
        /* Skill tag styling */
        .skill-tag {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 15px;
            font-size: 0.85rem;
            margin: 0.2rem;
            font-weight: 500;
        }
        
        /* Strength/Gap item styling */
        .strength-item {
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%);
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        .gap-item {
            background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(220, 53, 69, 0.05) 100%);
            border-left: 4px solid #dc3545;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem 0;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Progress indicator */
        .progress-indicator {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        }
        
        /* Info box styling */
        .info-box {
            background: linear-gradient(135deg, rgba(23, 162, 184, 0.1) 0%, rgba(23, 162, 184, 0.05) 100%);
            border-left: 4px solid #17a2b8;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Metric value styling */
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            border: 2px dashed #667eea;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            padding: 2rem;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 1rem 2rem;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header() -> None:
    """Render the main header"""
    st.markdown("""
        <div class="main-header">
            <h1>🚀 Nova HR Assistant</h1>
            <p>AI-Powered Professional CV Analysis & Candidate Evaluation</p>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render enhanced sidebar"""
    with st.sidebar:
        st.markdown("## 🚀 Nova HR")
        st.caption("Next-generation CV analysis powered by AI")
        
        st.markdown("---")
        
        st.markdown("### 📋 Position Details")
        st.session_state.position_description = st.text_area(
            "Role description and requirements",
            placeholder="Paste the complete job description here...\n\nInclude:\n• Required skills\n• Experience level\n• Education requirements\n• Key responsibilities",
            height=200,
            help="Provide detailed job requirements for accurate candidate matching"
        )
        
        st.markdown("### 🎨 Theme")
        theme_options = {
            "Midnight": {"primary": "#667eea", "secondary": "#764ba2"},
            "Ocean": {"primary": "#17a2b8", "secondary": "#5bc0de"},
            "Forest": {"primary": "#28a745", "secondary": "#20c997"},
            "Sunset": {"primary": "#fd7e14", "secondary": "#ffc107"}
        }
        
        theme = st.selectbox(
            "Color theme",
            options=list(theme_options.keys()),
            index=0,
            help="Customize the visual appearance of charts and components"
        )
        st.session_state.theme = theme
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.analysis_result:
            st.markdown("### 📊 Quick Stats")
            result = st.session_state.analysis_result
            score = result.get("evaluation", {}).get("suitability_score", 0)
            
            st.metric("Suitability Score", f"{score}/100")
            st.metric("Confidence", f"{int(result.get('evaluation', {}).get('scoring_details', {}).get('confidence_level', 0) * 100)}%")
        
        st.markdown("---")
        st.caption("Made with ❤️ by Nova | Powered by Advanced AI")


def upload_section() -> Optional[str]:
    """Render file upload section with enhanced UI"""
    st.markdown('<div class="section-header">📄 Upload Candidate CV</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader(
            "Drag and drop or click to upload",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            help="Supported formats: PDF, DOCX, TXT"
        )
    
    with col2:
        st.markdown("""
            <div class="info-box">
                <strong>📝 Supported Formats:</strong><br>
                • PDF (.pdf)<br>
                • Word (.docx)<br>
                • Text (.txt)
            </div>
        """, unsafe_allow_html=True)
    
    cv_text: Optional[str] = None
    if uploaded is not None:
        st.session_state.uploaded_file_name = uploaded.name
        
        with st.spinner("📖 Parsing document..."):
            temp_path = Path(".tmp_upload_" + uploaded.name)
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                cv_text = FileParser.extract_text(str(temp_path))
                st.success(f"✅ Successfully parsed: **{uploaded.name}**")
            except FileParsingError as e:
                st.error(f"❌ File parsing failed: {str(e)}")
            finally:
                temp_path.unlink(missing_ok=True)
    
    return cv_text


def create_enhanced_timeline(step: int, total: int) -> None:
    """Create beautiful animated timeline"""
    steps = [
        {"name": "Parse CV", "icon": "📄"},
        {"name": "Experience", "icon": "💼"},
        {"name": "Skills", "icon": "🎯"},
        {"name": "Education", "icon": "🎓"},
        {"name": "Certifications", "icon": "📜"},
        {"name": "Scoring", "icon": "⭐"},
        {"name": "Report", "icon": "📊"}
    ]
    
    theme_colors = {
        "Midnight": "#667eea",
        "Ocean": "#17a2b8",
        "Forest": "#28a745",
        "Sunset": "#fd7e14"
    }
    
    accent = theme_colors.get(st.session_state.theme, "#667eea")
    
    # Create progress indicators
    cols = st.columns(total)
    for i, (col, step_info) in enumerate(zip(cols, steps[:total])):
        with col:
            if i < step:
                status = "✅"
                opacity = 1.0
            elif i == step:
                status = "⏳"
                opacity = 0.8
            else:
                status = "⏸️"
                opacity = 0.3
            
            st.markdown(f"""
                <div style="text-align: center; opacity: {opacity};">
                    <div style="font-size: 2rem;">{status}</div>
                    <div style="font-size: 1.5rem;">{step_info['icon']}</div>
                    <div style="font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem;">
                        {step_info['name']}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def create_radar_chart(component_scores: Dict[str, float], theme: str) -> go.Figure:
    """Create radar chart for component scores"""
    theme_colors = {
        "Midnight": "#667eea",
        "Ocean": "#17a2b8",
        "Forest": "#28a745",
        "Sunset": "#fd7e14"
    }
    
    color = theme_colors.get(theme, "#667eea")
    
    categories = list(component_scores.keys())
    values = list(component_scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[c.title() for c in categories],
        fill='toself',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)',
        line=dict(color=color, width=3),
        name='Component Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_score_gauge(score: int, theme: str) -> go.Figure:
    """Create animated gauge chart for suitability score"""
    theme_colors = {
        "Midnight": "#667eea",
        "Ocean": "#17a2b8",
        "Forest": "#28a745",
        "Sunset": "#fd7e14"
    }
    
    color = theme_colors.get(theme, "#667eea")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Suitability Score", 'font': {'size': 24, 'color': color}},
        number={'font': {'size': 60, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': color},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(220, 53, 69, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(255, 193, 7, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(40, 167, 69, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Inter"}
    )
    
    return fig


def render_results(result: Dict[str, Any]) -> None:
    """Render comprehensive analysis results"""
    candidate = result.get("candidate", {})
    evaluation = result.get("evaluation", {})
    analysis = result.get("analysis", {})
    scoring_details = evaluation.get("scoring_details", {})
    
    # Summary Cards
    st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    score = evaluation.get("suitability_score", 0)
    score_class = "excellent" if score >= 80 else "good" if score >= 60 else "fair" if score >= 40 else "poor"
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Suitability Score</div>
                <div class="score-badge score-{score_class}">{score}/100</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rec = evaluation.get("recommendation", "Unknown")
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Recommendation</div>
                <div style="font-size: 1.2rem; font-weight: 600;">{rec}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence = scoring_details.get("confidence_level", 0.5)
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Confidence</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{int(confidence * 100)}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        potential = scoring_details.get("potential_score", score)
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Potential Score</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{potential}/100</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        risk_count = len(scoring_details.get("risk_factors", []))
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Risk Factors</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{risk_count}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Candidate Profile
    st.markdown('<div class="section-header">👤 Candidate Profile</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📝 Basic Information</h3>
                <p><strong>Name:</strong> {candidate.get('name', 'Unknown')}</p>
                <p><strong>Contact:</strong> {candidate.get('contact_info', 'N/A')}</p>
                <p><strong>Location:</strong> {candidate.get('location', 'N/A')}</p>
                <p><strong>Experience:</strong> {analysis.get('experience', {}).get('years_experience', 0)} years</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Score gauge
        st.plotly_chart(
            create_score_gauge(score, st.session_state.theme),
            use_container_width=True,
            config={'displayModeBar': False}
        )
    
    # Component Scores Radar
    component_scores = scoring_details.get("component_scores", {})
    if component_scores:
        st.markdown('<div class="section-header">🎯 Component Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.plotly_chart(
                create_radar_chart(component_scores, st.session_state.theme),
                use_container_width=True,
                config={'displayModeBar': False}
            )
        
        with col2:
            st.markdown("### Score Breakdown")
            for component, value in component_scores.items():
                score_class = "excellent" if value >= 80 else "good" if value >= 60 else "fair" if value >= 40 else "poor"
                st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 0.5rem;">
                        <strong>{component.title()}:</strong> 
                        <span class="score-badge score-{score_class}" style="font-size: 0.9rem; padding: 0.3rem 0.6rem;">
                            {value}/100
                        </span>
                    </div>
                """, unsafe_allow_html=True)
    
    # Strengths and Improvements
    st.markdown('<div class="section-header">💪 Strengths & Development Areas</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Key Strengths")
        for strength in evaluation.get("strengths", []):
            st.markdown(f'<div class="strength-item">• {strength}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Areas for Improvement")
        for gap in evaluation.get("areas_for_improvement", []):
            st.markdown(f'<div class="gap-item">• {gap}</div>', unsafe_allow_html=True)
    
    # Skills Visualization
    st.markdown('<div class="section-header">🎯 Skills Profile</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Technical Skills")
        skills_html = ""
        for skill in analysis.get("skills", {}).get("technical_skills", [])[:15]:
            skills_html += f'<span class="skill-tag">{skill}</span>'
        st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Soft Skills")
        skills_html = ""
        for skill in analysis.get("skills", {}).get("soft_skills", [])[:15]:
            skills_html += f'<span class="skill-tag">{skill}</span>'
        st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
    
    # Detailed Analysis Tabs
    st.markdown('<div class="section-header">📋 Detailed Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💼 Experience",
        "🎯 Skills",
        "🎓 Education",
        "📜 Certifications",
        "⚠️ Risk Analysis"
    ])
    
    with tab1:
        exp = analysis.get("experience", {})
        st.markdown(f"**Summary:** {exp.get('summary', 'N/A')}")
        st.markdown(f"**Years of Experience:** {exp.get('years_experience', 0)}")
        
        if exp.get('strengths'):
            st.markdown("**Strengths:**")
            for s in exp['strengths']:
                st.markdown(f'<div class="strength-item">{s}</div>', unsafe_allow_html=True)
        
        if exp.get('gaps'):
            st.markdown("**Gaps:**")
            for g in exp['gaps']:
                st.markdown(f'<div class="gap-item">{g}</div>', unsafe_allow_html=True)
    
    with tab2:
        skills = analysis.get("skills", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Technical Skills:**")
            for skill in skills.get('technical_skills', []):
                st.markdown(f"• {skill}")
        
        with col2:
            st.markdown("**Soft Skills:**")
            for skill in skills.get('soft_skills', []):
                st.markdown(f"• {skill}")
        
        if skills.get('skill_gaps'):
            st.markdown("**Skill Gaps:**")
            for gap in skills['skill_gaps']:
                st.markdown(f'<div class="gap-item">{gap}</div>', unsafe_allow_html=True)
    
    with tab3:
        edu = analysis.get("education", {})
        st.markdown(f"**Summary:** {edu.get('summary', 'N/A')}")
        
        if edu.get('degrees'):
            st.markdown("**Degrees:**")
            for degree in edu['degrees']:
                st.markdown(f"• {degree}")
    
    with tab4:
        certs = analysis.get("certifications", {})
        
        if certs.get('certifications'):
            st.markdown("**Certifications:**")
            for cert in certs['certifications']:
                st.markdown(f"• {cert}")
        else:
            st.info("No certifications listed")
    
    with tab5:
        risk_factors = scoring_details.get("risk_factors", [])
        
        if risk_factors:
            st.warning(f"**{len(risk_factors)} Risk Factors Identified:**")
            for risk in risk_factors:
                st.markdown(f'<div class="gap-item">{risk}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ No significant risk factors identified")
        
        if scoring_details.get('development_timeline'):
            st.info(f"**Development Timeline:** {scoring_details['development_timeline']}")


def run_analysis(cv_text: str, position: str) -> Dict[str, Any]:
    """Execute CV analysis pipeline"""
    return st.session_state.nova.process_candidate(cv_text, position)


def main() -> None:
    """Main application entry point"""
    set_page_config()
    inject_custom_css()
    init_state()
    
    render_header()
    render_sidebar()
    
    # Upload section
    cv_text = upload_section()
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        can_analyze = cv_text is not None and st.session_state.position_description.strip() != ""
        analyze_btn = st.button(
            "🚀 Analyze Candidate",
            type="primary",
            use_container_width=True,
            disabled=not can_analyze
        )
    
    with col2:
        if st.session_state.uploaded_file_name:
            st.success(f"📄 Ready: {st.session_state.uploaded_file_name}")
        else:
            st.info("📤 Upload a CV to begin")
    
    with col3:
        if st.session_state.analysis_result:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.analysis_result = None
                st.session_state.uploaded_file_name = None
                st.rerun()
    
    # Analysis execution
    if analyze_btn and cv_text:
        st.markdown('<div class="section-header">⚙️ Analysis Pipeline</div>', unsafe_allow_html=True)
        
        # Progress tracking
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        steps_total = 7
        
        for i in range(steps_total + 1):
            st.session_state.progress_step = i
            
            with progress_placeholder.container():
                create_enhanced_timeline(i, steps_total)
            
            if i < steps_total:
                step_names = [
                    "Parsing CV document",
                    "Analyzing work experience",
                    "Evaluating skills profile",
                    "Reviewing education background",
                    "Checking certifications",
                    "Calculating suitability score",
                    "Compiling comprehensive report"
                ]
                
                with status_placeholder:
                    st.info(f"⏳ {step_names[i]}...")
            
            if i == steps_total - 1:
                with st.spinner("🔍 Performing deep analysis..."):
                    try:
                        result = run_analysis(cv_text, st.session_state.position_description)
                        
                        # Add metadata
                        result['metadata'] = {
                            'position': st.session_state.position_description[:100],
                            'analysis_timestamp': datetime.now().isoformat(),
                            'filename': st.session_state.uploaded_file_name
                        }
                        
                        st.session_state.analysis_result = result
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.stop()
        
        status_placeholder.success("✅ Analysis complete!")
    
    # Display results
    if st.session_state.analysis_result:
        st.markdown("---")
        render_results(st.session_state.analysis_result)
        
        # Save results
        st.markdown("---")
        st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            RESULTS_DIR.mkdir(exist_ok=True)
            candidate_name = st.session_state.analysis_result.get('candidate', {}).get('name', 'candidate')
            safe_name = "".join(c for c in candidate_name if c.isalnum() or c in (' ', '-', '_')).strip()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_file = RESULTS_DIR / f"{safe_name}_{timestamp}.json"
            
            try:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.analysis_result, f, indent=2, ensure_ascii=False)
                st.success(f"✅ Saved to: `{out_file.name}`")
            except Exception as e:
                st.error(f"❌ Save failed: {str(e)}")
        
        with col2:
            # Download button
            json_str = json.dumps(st.session_state.analysis_result, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{safe_name}_analysis.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            st.info("💡 Use `visualize_candidates.py` to compare multiple candidates")


if __name__ == "__main__":
    main()