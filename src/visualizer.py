"""
Visualization dashboard and analytics for resume-job matching results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from typing import Dict, Any, List, Optional
import logging
from .bigquery_client import BigQueryAIClient
from .config import get_config

class Visualizer:
    """Create visualizations and dashboards for resume matching analytics"""
    
    def __init__(self):
        self.client = BigQueryAIClient()
        self.config = get_config('visualization')
        self.logger = logging.getLogger(__name__)
        
        # Set visualization style
        plt.style.use(self.config['style'])
        sns.set_palette(self.config['color_palette'])
    
    def create_match_distribution_chart(self, save_path: str = None) -> go.Figure:
        """Create distribution chart of match scores"""
        
        try:
            # Get match data
            query = f"""
            SELECT similarity_score
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}`
            """
            
            matches_df = self.client.client.query(query).to_dataframe()
            
            if matches_df.empty:
                self.logger.warning("No match data available for visualization")
                return go.Figure()
            
            # Create histogram
            fig = px.histogram(
                matches_df, 
                x='similarity_score',
                nbins=20,
                title='Distribution of Resume-Job Match Scores',
                labels={'similarity_score': 'Similarity Score', 'count': 'Number of Matches'},
                color_discrete_sequence=['#1f77b4']
            )
            
            fig.update_layout(
                xaxis_title="Similarity Score",
                yaxis_title="Number of Matches",
                showlegend=False,
                template="plotly_white"
            )
            
            # Add vertical line for threshold
            threshold = get_config('matching')['similarity_threshold']
            fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"Threshold: {threshold}")
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating match distribution chart: {e}")
            return go.Figure()
    
    def create_skills_analysis_chart(self, save_path: str = None) -> go.Figure:
        """Create skills demand analysis chart"""
        
        try:
            # Get skills data from job descriptions
            query = f"""
            SELECT required_skills
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}`
            WHERE required_skills IS NOT NULL
            """
            
            jobs_df = self.client.client.query(query).to_dataframe()
            
            if jobs_df.empty:
                return go.Figure()
            
            # Extract and count skills
            all_skills = []
            for skills_str in jobs_df['required_skills']:
                if skills_str:
                    skills = [skill.strip().title() for skill in skills_str.split(',')]
                    all_skills.extend(skills)
            
            # Count skill frequency
            skills_series = pd.Series(all_skills)
            top_skills = skills_series.value_counts().head(15)
            
            # Create bar chart
            fig = px.bar(
                x=top_skills.values,
                y=top_skills.index,
                orientation='h',
                title='Most In-Demand Skills',
                labels={'x': 'Number of Job Postings', 'y': 'Skills'},
                color=top_skills.values,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                template="plotly_white"
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating skills analysis chart: {e}")
            return go.Figure()
    
    def create_experience_vs_matches_chart(self, save_path: str = None) -> go.Figure:
        """Create scatter plot of experience vs match quality"""
        
        try:
            # Get combined data
            query = f"""
            SELECT 
                r.experience_years,
                m.similarity_score,
                j.experience_required
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}` m
            JOIN `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}` r 
                ON m.resume_id = r.resume_id
            JOIN `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}` j 
                ON m.job_id = j.job_id
            WHERE r.experience_years IS NOT NULL 
            AND j.experience_required IS NOT NULL
            """
            
            data_df = self.client.client.query(query).to_dataframe()
            
            if data_df.empty:
                return go.Figure()
            
            # Create scatter plot
            fig = px.scatter(
                data_df,
                x='experience_years',
                y='similarity_score',
                color='experience_required',
                title='Experience vs Match Quality',
                labels={
                    'experience_years': 'Candidate Experience (Years)',
                    'similarity_score': 'Match Score',
                    'experience_required': 'Required Experience'
                },
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(template="plotly_white")
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating experience vs matches chart: {e}")
            return go.Figure()
    
    def create_company_matches_chart(self, save_path: str = None) -> go.Figure:
        """Create chart showing matches by company"""
        
        try:
            query = f"""
            SELECT 
                j.company,
                COUNT(*) as match_count,
                AVG(m.similarity_score) as avg_score
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}` m
            JOIN `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}` j 
                ON m.job_id = j.job_id
            WHERE j.company IS NOT NULL
            GROUP BY j.company
            ORDER BY match_count DESC
            LIMIT 15
            """
            
            company_df = self.client.client.query(query).to_dataframe()
            
            if company_df.empty:
                return go.Figure()
            
            # Create subplot with two y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bar chart for match count
            fig.add_trace(
                go.Bar(
                    x=company_df['company'],
                    y=company_df['match_count'],
                    name="Match Count",
                    marker_color='lightblue'
                ),
                secondary_y=False,
            )
            
            # Add line chart for average score
            fig.add_trace(
                go.Scatter(
                    x=company_df['company'],
                    y=company_df['avg_score'],
                    mode='lines+markers',
                    name="Avg Match Score",
                    line=dict(color='red', width=3)
                ),
                secondary_y=True,
            )
            
            # Update layout
            fig.update_xaxes(title_text="Company", tickangle=45)
            fig.update_yaxes(title_text="Number of Matches", secondary_y=False)
            fig.update_yaxes(title_text="Average Match Score", secondary_y=True)
            fig.update_layout(
                title_text="Matches by Company",
                template="plotly_white"
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating company matches chart: {e}")
            return go.Figure()
    
    def create_skills_wordcloud(self, data_type: str = "jobs", save_path: str = None) -> None:
        """Create word cloud of skills"""
        
        try:
            if data_type == "jobs":
                query = f"""
                SELECT required_skills as skills
                FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}`
                WHERE required_skills IS NOT NULL
                """
            else:
                query = f"""
                SELECT skills
                FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}`
                WHERE skills IS NOT NULL
                """
            
            data_df = self.client.client.query(query).to_dataframe()
            
            if data_df.empty:
                return
            
            # Combine all skills text
            all_skills_text = ' '.join(data_df['skills'].dropna())
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(all_skills_text)
            
            # Plot
            plt.figure(figsize=self.config['figure_size'])
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Skills Word Cloud - {data_type.title()}', fontsize=16, pad=20)
            
            if save_path:
                plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating skills word cloud: {e}")
    
    def create_match_quality_pie_chart(self, save_path: str = None) -> go.Figure:
        """Create pie chart of match quality distribution"""
        
        try:
            query = f"""
            SELECT 
                CASE 
                    WHEN similarity_score >= 0.9 THEN 'Excellent (0.9+)'
                    WHEN similarity_score >= 0.8 THEN 'Very Good (0.8-0.9)'
                    WHEN similarity_score >= 0.7 THEN 'Good (0.7-0.8)'
                    WHEN similarity_score >= 0.6 THEN 'Fair (0.6-0.7)'
                    ELSE 'Poor (<0.6)'
                END as quality_category,
                COUNT(*) as count
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}`
            GROUP BY quality_category
            ORDER BY count DESC
            """
            
            quality_df = self.client.client.query(query).to_dataframe()
            
            if quality_df.empty:
                return go.Figure()
            
            # Create pie chart
            fig = px.pie(
                quality_df,
                values='count',
                names='quality_category',
                title='Match Quality Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(template="plotly_white")
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating match quality pie chart: {e}")
            return go.Figure()
    
    def create_comprehensive_dashboard(self, save_path: str = "dashboard.html") -> None:
        """Create comprehensive dashboard with multiple visualizations"""
        
        try:
            # Create all charts
            charts = {
                'match_distribution': self.create_match_distribution_chart(),
                'skills_analysis': self.create_skills_analysis_chart(),
                'experience_matches': self.create_experience_vs_matches_chart(),
                'company_matches': self.create_company_matches_chart(),
                'match_quality': self.create_match_quality_pie_chart()
            }
            
            # Create HTML dashboard
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI-Powered Resume Matcher - Analytics Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .chart-container {{ margin: 20px 0; }}
                    .row {{ display: flex; flex-wrap: wrap; }}
                    .col {{ flex: 1; min-width: 500px; margin: 10px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 AI-Powered Resume Matcher</h1>
                    <h2>Analytics Dashboard</h2>
                    <p>Comprehensive insights into resume-job matching performance</p>
                </div>
                
                <div class="row">
                    <div class="col">
                        <div id="match_distribution" class="chart-container"></div>
                    </div>
                    <div class="col">
                        <div id="match_quality" class="chart-container"></div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col">
                        <div id="skills_analysis" class="chart-container"></div>
                    </div>
                    <div class="col">
                        <div id="experience_matches" class="chart-container"></div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col" style="flex: 2;">
                        <div id="company_matches" class="chart-container"></div>
                    </div>
                </div>
                
                <script>
            """
            
            # Add chart scripts
            for chart_id, fig in charts.items():
                if fig.data:  # Only add if chart has data
                    html_content += f"""
                    Plotly.newPlot('{chart_id}', {fig.to_json()});
                    """
            
            html_content += """
                </script>
            </body>
            </html>
            """
            
            # Save dashboard
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Dashboard saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive dashboard: {e}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary analytics report"""
        
        report = {
            'total_resumes': 0,
            'total_jobs': 0,
            'total_matches': 0,
            'avg_match_score': 0,
            'top_skills': [],
            'match_quality_breakdown': {},
            'recommendations': []
        }
        
        try:
            # Get basic counts
            counts_query = f"""
            SELECT 
                (SELECT COUNT(*) FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}`) as total_resumes,
                (SELECT COUNT(*) FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}`) as total_jobs,
                (SELECT COUNT(*) FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}`) as total_matches,
                (SELECT AVG(similarity_score) FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}`) as avg_score
            """
            
            counts_result = self.client.client.query(counts_query).to_dataframe()
            
            if not counts_result.empty:
                row = counts_result.iloc[0]
                report['total_resumes'] = int(row['total_resumes'])
                report['total_jobs'] = int(row['total_jobs'])
                report['total_matches'] = int(row['total_matches'])
                report['avg_match_score'] = float(row['avg_score']) if row['avg_score'] else 0
            
            # Generate recommendations based on data
            if report['avg_match_score'] < 0.7:
                report['recommendations'].append("Consider expanding the skills database or improving text preprocessing")
            
            if report['total_matches'] < report['total_resumes'] * 0.5:
                report['recommendations'].append("Low match rate detected - review similarity threshold settings")
            
            report['recommendations'].append("Regular model retraining recommended for optimal performance")
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
        
        return report
