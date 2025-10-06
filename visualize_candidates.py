#!/usr/bin/env python3
"""
Candidate Comparison Visualization Tool for Nova HR Assistant

This script analyzes JSON analysis results from the results folder and creates
a comprehensive HTML comparison page with rankings, charts, and detailed insights.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


def setup_logging(debug_mode: bool = False) -> None:
    """Set up logging for visualization tool"""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler('nova_visualization.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_candidate_data(results_folder: str) -> List[Dict[str, Any]]:
    """
    Load all candidate analysis results from the results folder.
    
    Args:
        results_folder: Path to the results folder
        
    Returns:
        List of candidate data dictionaries
    """
    results_path = Path(results_folder)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results folder not found: {results_folder}")
    
    if not results_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {results_folder}")
    
    candidates = []
    
    # Find all JSON files in results folder
    json_files = list(results_path.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON analysis files found in {results_folder}")
    
    print(f"📁 Found {len(json_files)} analysis files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate basic structure
            if not all(key in data for key in ['candidate', 'evaluation', 'analysis']):
                print(f"⚠️  Skipping {json_file.name}: Invalid structure")
                continue
            
            # Add filename for reference
            data['filename'] = json_file.name
            candidates.append(data)
            
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {str(e)}")
            continue
    
    if not candidates:
        raise ValueError("No valid candidate data found")
    
    print(f"✅ Loaded {len(candidates)} valid candidate analyses")
    return candidates


def extract_comparison_data(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract data for comparison from candidate analyses.
    
    Args:
        candidates: List of candidate data dictionaries
        
    Returns:
        Dictionary containing comparison data
    """
    comparison_data = {
        'candidates': [],
        'position': None,
        'analysis_date': None,
        'total_candidates': len(candidates)
    }
    
    for candidate in candidates:
        # Basic candidate info
        candidate_info = candidate.get('candidate', {})
        evaluation = candidate.get('evaluation', {})
        analysis = candidate.get('analysis', {})
        
        # Extract key metrics including enhanced scoring data
        scoring_details = evaluation.get('scoring_details', {})
        
        candidate_data = {
            'name': candidate_info.get('name', 'Unknown'),
            'filename': candidate.get('filename', ''),
            'suitability_score': evaluation.get('suitability_score', 0),
            'recommendation': evaluation.get('recommendation', 'Unknown'),
            'strengths': evaluation.get('strengths', []),
            'areas_for_improvement': evaluation.get('areas_for_improvement', []),
            'contact_info': candidate_info.get('contact_info', ''),
            'location': candidate_info.get('location', ''),
            
            # Enhanced scoring metrics
            'confidence_level': scoring_details.get('confidence_level', 0.5),
            'risk_factors': scoring_details.get('risk_factors', []),
            'potential_score': scoring_details.get('potential_score', evaluation.get('suitability_score', 0)),
            'development_timeline': scoring_details.get('development_timeline', ''),
            'component_scores': scoring_details.get('component_scores', {}),
            'scoring_weights': scoring_details.get('scoring_weights', {}),
            'bias_detection': scoring_details.get('bias_detection', None),
            
            # Experience data
            'years_experience': analysis.get('experience', {}).get('years_experience', 0),
            'experience_summary': analysis.get('experience', {}).get('summary', ''),
            'experience_strengths': analysis.get('experience', {}).get('strengths', []),
            'experience_gaps': analysis.get('experience', {}).get('gaps', []),
            
            # Skills data
            'technical_skills': analysis.get('skills', {}).get('technical_skills', []),
            'soft_skills': analysis.get('skills', {}).get('soft_skills', []),
            'skill_gaps': analysis.get('skills', {}).get('skill_gaps', []),
            'proficiency_levels': analysis.get('skills', {}).get('proficiency_levels', {}),
            
            # Education data
            'education_summary': analysis.get('education', {}).get('summary', ''),
            'degrees': analysis.get('education', {}).get('degrees', []),
            'education_strengths': analysis.get('education', {}).get('strengths', []),
            'education_gaps': analysis.get('education', {}).get('gaps', []),
            
            # Certifications data
            'certifications': analysis.get('certifications', {}).get('certifications', []),
            'certification_strengths': analysis.get('certifications', {}).get('strengths', []),
            'certification_gaps': analysis.get('certifications', {}).get('gaps', []),
            
            # Detailed analysis
            'detailed_strengths': evaluation.get('detailed_strengths_analysis', ''),
            'detailed_weaknesses': evaluation.get('detailed_weaknesses_analysis', ''),
            'comprehensive_profile': evaluation.get('comprehensive_candidate_profile', ''),
            
            # Summary data
            'overall_assessment': candidate.get('summary', {}).get('overall_assessment', ''),
            'key_highlights': candidate.get('summary', {}).get('key_highlights', []),
            'decision_factors': candidate.get('summary', {}).get('decision_factors', [])
        }
        
        comparison_data['candidates'].append(candidate_data)
        
        # Set position and analysis date from first candidate
        if comparison_data['position'] is None:
            comparison_data['position'] = candidate.get('metadata', {}).get('position', 'Unknown Position')
            comparison_data['analysis_date'] = candidate.get('metadata', {}).get('analysis_timestamp', '')
    
    # Sort candidates by suitability score (highest first)
    comparison_data['candidates'].sort(key=lambda x: x['suitability_score'], reverse=True)
    
    return comparison_data


def generate_html_report(comparison_data: Dict[str, Any], output_file: str) -> None:
    """
    Generate comprehensive HTML comparison report.
    
    Args:
        comparison_data: Extracted comparison data
        output_file: Output HTML file path
    """
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nova HR - Candidate Comparison Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .ranking-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .ranking-table th,
        .ranking-table td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .ranking-table th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        .ranking-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .ranking-table tr:hover {{
            background-color: #f0f0f0;
        }}
        .score-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .score-excellent {{ background-color: #28a745; }}
        .score-good {{ background-color: #17a2b8; }}
        .score-fair {{ background-color: #ffc107; color: #333; }}
        .score-poor {{ background-color: #dc3545; }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .candidate-detail {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .candidate-detail h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .skills-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0;
        }}
        .skill-tag {{
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .strength-item, .gap-item {{
            margin: 8px 0;
            padding: 8px 12px;
            border-radius: 5px;
        }}
        .strength-item {{
            background: #d4edda;
            border-left: 3px solid #28a745;
        }}
        .gap-item {{
            background: #f8d7da;
            border-left: 3px solid #dc3545;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }}
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
                border-radius: 5px;
            }}
            .header {{
                padding: 20px;
            }}
            .header h1 {{
                font-size: 2em;
            }}
            .section {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Nova HR Assistant</h1>
            <p>Candidate Comparison Report</p>
            <p><strong>Position:</strong> {comparison_data['position']}</p>
            <p><strong>Analysis Date:</strong> {comparison_data['analysis_date'][:10] if comparison_data['analysis_date'] else 'Unknown'}</p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-number">{comparison_data['total_candidates']}</div>
                <div class="stat-label">Total Candidates</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{max(c['suitability_score'] for c in comparison_data['candidates'])}</div>
                <div class="stat-label">Highest Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{sum(c['suitability_score'] for c in comparison_data['candidates']) / len(comparison_data['candidates']):.1f}</div>
                <div class="stat-label">Average Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len([c for c in comparison_data['candidates'] if c['suitability_score'] >= 70])}</div>
                <div class="stat-label">Strong Candidates (70+)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{sum(c['confidence_level'] for c in comparison_data['candidates']) / len(comparison_data['candidates']):.2f}</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len([c for c in comparison_data['candidates'] if c['potential_score'] > c['suitability_score']])}</div>
                <div class="stat-label">High Potential</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Candidate Ranking</h2>
            <table class="ranking-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Name</th>
                        <th>Suitability Score</th>
                        <th>Potential Score</th>
                        <th>Confidence</th>
                        <th>Risk Factors</th>
                        <th>Recommendation</th>
                        <th>Years Experience</th>
                        <th>Key Strengths</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add ranking table rows
    for i, candidate in enumerate(comparison_data['candidates'], 1):
        score_class = 'score-excellent' if candidate['suitability_score'] >= 80 else \
                     'score-good' if candidate['suitability_score'] >= 60 else \
                     'score-fair' if candidate['suitability_score'] >= 40 else 'score-poor'
        
        potential_class = 'score-excellent' if candidate['potential_score'] >= 80 else \
                         'score-good' if candidate['potential_score'] >= 60 else \
                         'score-fair' if candidate['potential_score'] >= 40 else 'score-poor'
        
        confidence_class = 'score-excellent' if candidate['confidence_level'] >= 0.8 else \
                          'score-good' if candidate['confidence_level'] >= 0.6 else \
                          'score-fair' if candidate['confidence_level'] >= 0.4 else 'score-poor'
        
        risk_count = len(candidate['risk_factors'])
        risk_display = f"{risk_count} risks" if risk_count > 0 else "Low risk"
        
        top_strengths = candidate['strengths'][:2] if candidate['strengths'] else ['No strengths listed']
        
        html_content += f"""
                    <tr>
                        <td><strong>#{i}</strong></td>
                        <td>{candidate['name']}</td>
                        <td><span class="score-badge {score_class}">{candidate['suitability_score']}/100</span></td>
                        <td><span class="score-badge {potential_class}">{candidate['potential_score']}/100</span></td>
                        <td><span class="score-badge {confidence_class}">{candidate['confidence_level']:.2f}</span></td>
                        <td>{risk_display}</td>
                        <td>{candidate['recommendation']}</td>
                        <td>{candidate['years_experience']}</td>
                        <td>{', '.join(top_strengths)}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Suitability Score Comparison</h2>
            <div class="chart-container">
                <canvas id="scoreChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Enhanced Scoring Analysis</h2>
            <div class="chart-container">
                <canvas id="enhancedChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <div class="section">
            <h2>⚠️ Risk Assessment</h2>
            <div class="comparison-grid">
"""
    
    # Add risk assessment for each candidate
    for candidate in comparison_data['candidates']:
        risk_count = len(candidate['risk_factors'])
        risk_class = 'score-excellent' if risk_count == 0 else \
                    'score-good' if risk_count <= 2 else \
                    'score-fair' if risk_count <= 4 else 'score-poor'
        
        html_content += f"""
                <div class="metric-card">
                    <div class="metric-title">{candidate['name']}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(100, (5 - risk_count) * 20)}%"></div>
                    </div>
                    <p><strong>Risk Level:</strong> <span class="score-badge {risk_class}">{risk_count} risk factors</span></p>
                    <p><strong>Confidence:</strong> {candidate['confidence_level']:.2f}</p>
                    <p><strong>Potential Score:</strong> {candidate['potential_score']}/100</p>
                    <p><strong>Development Timeline:</strong> {candidate['development_timeline'] or 'Not specified'}</p>
                    <h4>Risk Factors:</h4>
"""
        for risk in candidate['risk_factors'][:3]:  # Show top 3 risks
            html_content += f'<div class="gap-item">{risk}</div>'
        
        html_content += """
                </div>
"""
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Component Score Analysis</h2>
            <div class="chart-container">
                <canvas id="componentChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        
        <div class="section">
            <h2>🎯 Skills Analysis</h2>
            <div class="comparison-grid">
"""
    
    # Add skills comparison for each candidate
    for candidate in comparison_data['candidates']:
        html_content += f"""
                <div class="metric-card">
                    <div class="metric-title">{candidate['name']}</div>
                    <div class="skills-list">
"""
        for skill in candidate['technical_skills'][:10]:  # Limit to top 10 skills
            html_content += f'<span class="skill-tag">{skill}</span>'
        
        html_content += """
                    </div>
                </div>
"""
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>👥 Detailed Candidate Analysis</h2>
"""
    
    # Add detailed analysis for each candidate
    for candidate in comparison_data['candidates']:
        html_content += f"""
            <div class="candidate-detail">
                <h3>{candidate['name']} - Score: {candidate['suitability_score']}/100</h3>
                <p><strong>Recommendation:</strong> {candidate['recommendation']}</p>
                <p><strong>Experience:</strong> {candidate['years_experience']} years</p>
                <p><strong>Location:</strong> {candidate['location']}</p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                    <div class="metric-card">
                        <div class="metric-title">Current Score</div>
                        <div class="stat-number">{candidate['suitability_score']}/100</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Potential Score</div>
                        <div class="stat-number">{candidate['potential_score']}/100</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Confidence Level</div>
                        <div class="stat-number">{candidate['confidence_level']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Risk Factors</div>
                        <div class="stat-number">{len(candidate['risk_factors'])}</div>
                    </div>
                </div>
                
                <h4>💪 Key Strengths:</h4>
"""
        for strength in candidate['strengths']:
            html_content += f'<div class="strength-item">{strength}</div>'
        
        html_content += """
                <h4>⚠️ Areas for Improvement:</h4>
"""
        for gap in candidate['areas_for_improvement']:
            html_content += f'<div class="gap-item">{gap}</div>'
        
        html_content += f"""
                <h4>🎓 Education:</h4>
                <p>{candidate['education_summary']}</p>
                
                <h4>📜 Certifications:</h4>
                <p>{', '.join(candidate['certifications']) if candidate['certifications'] else 'No certifications listed'}</p>
                
                <h4>🎯 Enhanced Metrics:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0;">
                    <div class="metric-card">
                        <div class="metric-title">Development Timeline</div>
                        <p>{candidate['development_timeline'] or 'Not specified'}</p>
                    </div>
                </div>
                
                <h4>📊 Detailed Score Breakdown:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                    <div class="metric-card">
                        <div class="metric-title">Current Score</div>
                        <div class="stat-number">{candidate['suitability_score']}/100</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Potential Score</div>
                        <div class="stat-number">{candidate['potential_score']}/100</div>
                    </div>
                </div>
                
                <h4>🔍 Component Score Analysis:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
                    {_generate_component_score_cards(candidate)}
                </div>
"""
        
        # Add bias detection information if available
        if candidate['bias_detection']:
            bias_info = candidate['bias_detection']
            if bias_info.get('bias_detected', False):
                html_content += f"""
                <h4>🛡️ Bias Detection:</h4>
                <div class="gap-item">
                    <strong>Bias Detected:</strong> {bias_info.get('total_biases', 0)} potential biases identified
                </div>
                <div class="gap-item">
                    <strong>Severity:</strong> {bias_info.get('severity_distribution', {})}
                </div>
                <div class="gap-item">
                    <strong>Mitigation Applied:</strong> Bias mitigation strategies have been applied to ensure fair evaluation
                </div>
"""
            else:
                html_content += """
                <h4>🛡️ Bias Detection:</h4>
                <div class="strength-item">
                    <strong>No Biases Detected:</strong> Analysis appears to be free from significant biases
                </div>
"""
        
        html_content += """
            </div>
"""
    
    html_content += f"""
        </div>
        
        <div class="footer">
            <p>Generated by Nova HR Assistant on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>This report compares {comparison_data['total_candidates']} candidates for the position: {comparison_data['position']}</p>
        </div>
    </div>
    
    <script>
        // Suitability Score Chart
        const scoreCtx = document.getElementById('scoreChart').getContext('2d');
        new Chart(scoreCtx, {{
            type: 'bar',
            data: {{
                labels: {[f'"{c["name"]}"' for c in comparison_data['candidates']]},
                datasets: [{{
                    label: 'Suitability Score',
                    data: {[c['suitability_score'] for c in comparison_data['candidates']]},
                    backgroundColor: [
                        {', '.join([f'"rgba(102, 126, 234, {0.3 + (i/len(comparison_data["candidates"])) * 0.7})"' for i in range(len(comparison_data['candidates']))])}
                    ],
                    borderColor: [
                        {', '.join([f'"rgba(102, 126, 234, 1)"' for _ in range(len(comparison_data['candidates']))])}
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Candidate Suitability Scores'
                    }}
                }}
            }}
        }});
        
        // Enhanced Scoring Chart
        const enhancedCtx = document.getElementById('enhancedChart').getContext('2d');
        new Chart(enhancedCtx, {{
            type: 'line',
            data: {{
                labels: {[f'"{c["name"]}"' for c in comparison_data['candidates']]},
                datasets: [{{
                    label: 'Current Score',
                    data: {[c['suitability_score'] for c in comparison_data['candidates']]},
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: false
                }}, {{
                    label: 'Potential Score',
                    data: {[c['potential_score'] for c in comparison_data['candidates']]},
                    borderColor: 'rgba(118, 75, 162, 1)',
                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                    borderWidth: 3,
                    fill: false
                }}, {{
                    label: 'Confidence Level (×100)',
                    data: {[c['confidence_level'] * 100 for c in comparison_data['candidates']]},
                    borderColor: 'rgba(40, 167, 69, 1)',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 2,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Enhanced Scoring Analysis'
                    }}
                }}
            }}
        }});
        
        // Component Score Chart
        const componentCtx = document.getElementById('componentChart').getContext('2d');
        new Chart(componentCtx, {{
            type: 'radar',
            data: {{
                labels: ['Experience', 'Skills', 'Education', 'Certifications', 'Overall Fit'],
                datasets: [
                    {', '.join([f'''{{
                        label: '{candidate['name']}',
                        data: [
                            {candidate['component_scores'].get('experience', 0)},
                            {candidate['component_scores'].get('skills', 0)},
                            {candidate['component_scores'].get('education', 0)},
                            {candidate['component_scores'].get('certifications', 0)},
                            {candidate['component_scores'].get('overall_fit', 0)}
                        ],
                        borderColor: 'rgba({102 + (i * 30) % 155}, {126 + (i * 40) % 129}, {234 + (i * 20) % 21}, 1)',
                        backgroundColor: 'rgba({102 + (i * 30) % 155}, {126 + (i * 40) % 129}, {234 + (i * 20) % 21}, 0.2)',
                        borderWidth: 2
                    }}''' for i, candidate in enumerate(comparison_data['candidates'])])}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Component Score Comparison'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_file}")


def _generate_component_score_cards(candidate: Dict[str, Any]) -> str:
    """
    Generate HTML cards for component scores with weights and contributions.
    
    Args:
        candidate: Candidate data dictionary
        
    Returns:
        HTML string with component score cards
    """
    component_scores = candidate.get('component_scores', {})
    scoring_weights = candidate.get('scoring_weights', {})
    
    if not component_scores:
        return '<div class="metric-card"><div class="metric-title">No Component Scores Available</div></div>'
    
    cards_html = ""
    
    for component, score in component_scores.items():
        weight = scoring_weights.get(component, 0.0)
        weighted_contribution = score * weight
        
        # Determine score color class
        if score >= 80:
            score_class = "score-excellent"
        elif score >= 60:
            score_class = "score-good"
        elif score >= 40:
            score_class = "score-fair"
        else:
            score_class = "score-poor"
        
        cards_html += f"""
                    <div class="metric-card">
                        <div class="metric-title">{component.title()}</div>
                        <div class="stat-number {score_class}">{score}/100</div>
                        <div style="margin-top: 10px;">
                            <small><strong>Weight:</strong> {weight:.1%}</small><br>
                            <small><strong>Contribution:</strong> {weighted_contribution:.1f}</small>
                        </div>
                    </div>"""
    
    return cards_html


def main() -> int:
    """Main application entry point"""
    
    parser = argparse.ArgumentParser(description='Generate candidate comparison visualization')
    parser.add_argument('--results-folder', default='results', 
                       help='Path to results folder (default: results)')
    parser.add_argument('--output', default='candidate_comparison.html',
                       help='Output HTML file (default: candidate_comparison.html)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        print("🚀 Nova HR Assistant - Candidate Comparison Tool")
        print("=" * 60)
        
        # Load candidate data
        print(f"📁 Loading candidate data from: {args.results_folder}")
        candidates = load_candidate_data(args.results_folder)
        
        # Extract comparison data
        print("🔍 Extracting comparison data...")
        comparison_data = extract_comparison_data(candidates)
        
        # Generate HTML report
        print(f"📊 Generating HTML report: {args.output}")
        generate_html_report(comparison_data, args.output)
        
        print("\n✅ Candidate comparison report generated successfully!")
        print(f"📄 Open {args.output} in your browser to view the results")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.error(f"Visualization error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

