"""
Vector View AI Financial Intelligence Dashboard

A comprehensive Streamlit dashboard for visualizing multi-agent financial analysis
including economic indicators, market intelligence, news sentiment, and editorial synthesis.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, Optional

# Import Vector View agents
from agents.orchestration_agent import OrchestrationAgent
from agents.base_agent import AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Vector View - AI Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class VectorViewDashboard:
    """Main dashboard class for Vector View financial intelligence"""
    
    def __init__(self):
        self.orchestrator = None
        self._initialize_orchestrator()
    
    def _initialize_orchestrator(self):
        """Initialize the orchestration agent"""
        try:
            database_url = "postgresql://postgres:fred_password@localhost:5432/postgres"
            self.orchestrator = OrchestrationAgent(database_url=database_url)
            # Auto-register all available agents
            self.orchestrator.register_all_agents()
            logger.info("Orchestration agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            st.error(f"Failed to initialize AI agents: {str(e)}")
    
    def render_header(self):
        """Render the main dashboard header"""
        st.markdown('<h1 class="main-header">🧠 Vector View AI Financial Intelligence</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card"><h3>📊 Economic Agent</h3><p>Active</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><h3>📰 News Sentiment</h3><p>Active</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card"><h3>📈 Market Intelligence</h3><p>Active</p></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card"><h3>✍️ Editorial Synthesis</h3><p>Active</p></div>', unsafe_allow_html=True)
    
    def render_query_interface(self):
        """Render the query input interface"""
        st.markdown("## 🔍 Financial Intelligence Query")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your financial analysis query:",
                placeholder="e.g., 'What's driving today's market movements?' or 'Analyze the correlation between inflation and tech stocks'"
            )
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe:",
                ["1d", "1w", "1m", "3m", "6m", "1y"],
                index=2
            )
        
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
        
        return query, timeframe, analyze_button
    
    def render_analysis_results(self, synthesis_result):
        """Render the comprehensive analysis results"""
        if not synthesis_result:
            return
        
        # Executive Summary
        st.markdown("## 📋 Executive Summary")
        st.markdown(f'<div class="agent-card">{synthesis_result.executive_summary}</div>', unsafe_allow_html=True)
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence_class = self._get_confidence_class(synthesis_result.confidence)
            st.markdown(f'<div class="metric-card"><h4>Overall Confidence</h4><p class="{confidence_class}">{synthesis_result.confidence:.1%}</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h4>Agents Executed</h4><p>{len(synthesis_result.agents_executed)}</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card"><h4>Execution Time</h4><p>{synthesis_result.total_execution_time_ms/1000:.1f}s</p></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-card"><h4>Cross-Domain Signals</h4><p>{len(synthesis_result.cross_domain_signals)}</p></div>', unsafe_allow_html=True)
        
        # Individual Agent Results
        st.markdown("## 🤖 Agent Analysis Results")
        
        # Create tabs for each agent
        agent_tabs = st.tabs([f"{agent.title()}" for agent in synthesis_result.agents_executed])
        
        for i, (agent_name, tab) in enumerate(zip(synthesis_result.agents_executed, agent_tabs)):
            with tab:
                self._render_agent_analysis(agent_name, synthesis_result.agent_responses.get(agent_name))
        
        # Key Insights
        st.markdown("## 💡 Key Insights")
        for i, insight in enumerate(synthesis_result.key_insights[:10], 1):
            st.markdown(f"**{i}.** {insight}")
        
        # Risk Assessment
        st.markdown("## ⚠️ Risk Assessment")
        risk_data = synthesis_result.risk_assessment
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Overall Risk Level:** {risk_data.get('overall_risk', 'Unknown').title()}")
            st.markdown(f"**Confidence Level:** {risk_data.get('confidence_level', 'Unknown').title()}")
        
        with col2:
            if risk_data.get('risk_factors'):
                st.markdown("**Risk Factors:**")
                for factor in risk_data['risk_factors'][:5]:
                    st.markdown(f"• {factor}")
    
    def _render_agent_analysis(self, agent_name: str, agent_response):
        """Render individual agent analysis results"""
        if not agent_response:
            st.warning(f"No data available for {agent_name}")
            return
        
        # Agent confidence and metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_class = self._get_confidence_class(agent_response.confidence)
            st.markdown(f'<p class="{confidence_class}">Confidence: {agent_response.confidence:.1%}</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Execution Time:** {agent_response.execution_time_ms:.1f}ms")
        
        with col3:
            st.markdown(f"**Data Sources:** {len(agent_response.data_sources_used)}")
        
        # Agent insights
        st.markdown("**Key Insights:**")
        for insight in agent_response.insights[:5]:
            st.markdown(f"• {insight}")
        
        # Key metrics visualization
        if agent_response.key_metrics:
            st.markdown("**Key Metrics:**")
            metrics_df = pd.DataFrame(list(agent_response.key_metrics.items()), columns=['Metric', 'Value'])
            
            # Create a simple bar chart for numeric metrics
            numeric_metrics = []
            for metric, value in agent_response.key_metrics.items():
                try:
                    numeric_value = float(value)
                    if -10 <= numeric_value <= 10:  # Reasonable range for visualization
                        numeric_metrics.append({'Metric': metric, 'Value': numeric_value})
                except (ValueError, TypeError):
                    pass
            
            if numeric_metrics:
                metrics_chart_df = pd.DataFrame(numeric_metrics)
                fig = px.bar(metrics_chart_df, x='Metric', y='Value', title=f"{agent_name.title()} Key Metrics")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(metrics_df, use_container_width=True)
        
        # Analysis details (expandable)
        with st.expander("View Detailed Analysis"):
            st.json(agent_response.analysis)
    
    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class for confidence level"""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def render_system_status(self):
        """Render system status in sidebar"""
        st.sidebar.markdown("## 🔧 System Status")
        
        if self.orchestrator:
            workflow_status = self.orchestrator.get_workflow_status()
            
            st.sidebar.markdown("**Available Workflows:**")
            for workflow in workflow_status.get('available_workflows', []):
                st.sidebar.markdown(f"✅ {workflow.replace('_', ' ').title()}")
            
            st.sidebar.markdown("**Registered Agents:**")
            for agent in workflow_status.get('registered_agents', []):
                st.sidebar.markdown(f"🤖 {agent.replace('_', ' ').title()}")
            
            # Performance stats
            perf_stats = workflow_status.get('performance_stats', {})
            if perf_stats:
                st.sidebar.markdown("**Performance:**")
                st.sidebar.markdown(f"• Queries: {perf_stats.get('total_queries', 0)}")
                st.sidebar.markdown(f"• Avg Time: {perf_stats.get('avg_execution_time_ms', 0):.1f}ms")
                st.sidebar.markdown(f"• Cache Hit Rate: {perf_stats.get('cache_hit_rate', 0):.1%}")
        else:
            st.sidebar.error("❌ Orchestrator not initialized")
    
    async def process_query_async(self, query: str, timeframe: str):
        """Process query asynchronously"""
        try:
            if not self.orchestrator:
                st.error("Orchestrator not available")
                return None
            
            # Process the query
            result = await self.orchestrator.process_user_query(
                query=query,
                timeframe=timeframe
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            st.error(f"Analysis failed: {str(e)}")
            return None
    
    def run(self):
        """Main dashboard execution"""
        # Render header
        self.render_header()
        
        # Render system status in sidebar
        self.render_system_status()
        
        # Render query interface
        query, timeframe, analyze_button = self.render_query_interface()
        
        # Process query if button clicked
        if analyze_button and query:
            with st.spinner("🧠 AI agents are analyzing your query..."):
                # Run async query processing
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.process_query_async(query, timeframe))
                    loop.close()
                    
                    if result:
                        # Store result in session state for persistence
                        st.session_state['last_result'] = result
                        st.session_state['last_query'] = query
                        st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Failed to process query: {str(e)}")
        
        # Display results if available
        if 'last_result' in st.session_state:
            st.markdown(f"### Results for: *{st.session_state.get('last_query', 'Previous Query')}*")
            self.render_analysis_results(st.session_state['last_result'])
        
        # Sample queries section
        st.markdown("## 💭 Sample Queries")
        sample_queries = [
            "What's driving today's market movements?",
            "Analyze the correlation between inflation and tech stocks",
            "Daily briefing on economic indicators",
            "How is news sentiment affecting market volatility?",
            "Deep dive analysis of Federal Reserve policy impact"
        ]
        
        cols = st.columns(len(sample_queries))
        for i, sample_query in enumerate(sample_queries):
            with cols[i]:
                if st.button(sample_query, key=f"sample_{i}"):
                    st.rerun()

def main():
    """Main application entry point"""
    dashboard = VectorViewDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
