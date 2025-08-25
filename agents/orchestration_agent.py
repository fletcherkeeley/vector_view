"""
Orchestration Agent for Vector View Financial Intelligence Platform

The master coordinator that manages query routing, agent coordination,
context assembly, and response synthesis across all specialist agents.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import json

from .base_agent import BaseAgent, AgentType, AgentContext, AgentResponse, ConfidenceLevel

logger = logging.getLogger(__name__)


@dataclass
class AgentWorkflow:
    """Defines the workflow for executing multiple agents"""
    workflow_id: str
    query_type: str
    required_agents: List[AgentType]
    execution_order: List[List[AgentType]]  # List of parallel execution groups
    dependencies: Dict[AgentType, List[AgentType]] = field(default_factory=dict)
    timeout_seconds: int = 300


@dataclass
class SynthesizedResponse:
    """Final synthesized response from multiple agents"""
    workflow_id: str
    query: str
    confidence: float
    
    # Aggregated insights
    executive_summary: str
    key_insights: List[str]
    cross_domain_signals: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    
    # Individual agent contributions
    agent_responses: Dict[str, AgentResponse]
    synthesis_metadata: Dict[str, Any]
    
    # Performance metrics
    total_execution_time_ms: float
    agents_executed: List[str]
    
    timestamp: datetime = field(default_factory=datetime.now)


class OrchestrationAgent(BaseAgent):
    """
    Master orchestration agent that coordinates all specialist agents.
    
    Responsibilities:
    - Query classification and routing
    - Context assembly and management
    - Agent workflow coordination
    - Response synthesis and conflict resolution
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        database_url: str,
        agent_registry: Optional[Dict[AgentType, BaseAgent]] = None,
        cache_ttl_minutes: int = 15,  # Shorter cache for orchestrator
        max_parallel_agents: int = 4
    ):
        super().__init__(
            agent_type=AgentType.ORCHESTRATION,
            database_url=database_url,
            cache_ttl_minutes=cache_ttl_minutes
        )
        
        # Agent registry for coordination
        self.agent_registry: Dict[AgentType, BaseAgent] = agent_registry or {}
        self.max_parallel_agents = max_parallel_agents
        
        # Workflow definitions
        self.workflows = self._initialize_workflows()
        
        # Cross-agent state management
        self.shared_state = {
            "market_regime": None,
            "economic_cycle": None,
            "risk_environment": None,
            "last_updated": None
        }
        
        logger.info("Orchestration agent initialized")
    
    def register_agent(self, agent: BaseAgent):
        """Register a specialist agent with the orchestrator"""
        self.agent_registry[agent.agent_type] = agent
        logger.info(f"Registered {agent.agent_type.value} agent")
    
    def _initialize_workflows(self) -> Dict[str, AgentWorkflow]:
        """Initialize predefined workflows for different query types"""
        return {
            "daily_briefing": AgentWorkflow(
                workflow_id="daily_briefing",
                query_type="daily_briefing",
                required_agents=[
                    AgentType.ECONOMIC,
                    AgentType.MARKET_INTELLIGENCE,
                    AgentType.NEWS_SENTIMENT,
                    AgentType.EDITORIAL_SYNTHESIS
                ],
                execution_order=[
                    [AgentType.ECONOMIC, AgentType.NEWS_SENTIMENT],  # First: Economic + Sentiment in parallel
                    [AgentType.MARKET_INTELLIGENCE],  # Second: Market Intelligence (needs sentiment data)
                    [AgentType.EDITORIAL_SYNTHESIS]  # Final: Synthesis
                ],
                dependencies={
                    AgentType.MARKET_INTELLIGENCE: [AgentType.NEWS_SENTIMENT],
                    AgentType.EDITORIAL_SYNTHESIS: [AgentType.ECONOMIC, AgentType.MARKET_INTELLIGENCE, AgentType.NEWS_SENTIMENT]
                },
                timeout_seconds=180
            ),
            
            "deep_dive": AgentWorkflow(
                workflow_id="deep_dive",
                query_type="deep_dive",
                required_agents=[
                    AgentType.ECONOMIC,
                    AgentType.MARKET_INTELLIGENCE,
                    AgentType.NEWS_SENTIMENT,
                    AgentType.EDITORIAL_SYNTHESIS
                ],
                execution_order=[
                    [AgentType.ECONOMIC, AgentType.NEWS_SENTIMENT],
                    [AgentType.MARKET_INTELLIGENCE],
                    [AgentType.EDITORIAL_SYNTHESIS]
                ],
                dependencies={
                    AgentType.MARKET_INTELLIGENCE: [AgentType.NEWS_SENTIMENT],
                    AgentType.EDITORIAL_SYNTHESIS: [AgentType.ECONOMIC, AgentType.MARKET_INTELLIGENCE, AgentType.NEWS_SENTIMENT]
                },
                timeout_seconds=300
            ),
            
            "correlation_analysis": AgentWorkflow(
                workflow_id="correlation_analysis",
                query_type="correlation_analysis",
                required_agents=[
                    AgentType.ECONOMIC,
                    AgentType.NEWS_SENTIMENT,
                    AgentType.EDITORIAL_SYNTHESIS
                ],
                execution_order=[
                    [AgentType.ECONOMIC, AgentType.NEWS_SENTIMENT],
                    [AgentType.EDITORIAL_SYNTHESIS]
                ],
                dependencies={
                    AgentType.EDITORIAL_SYNTHESIS: [AgentType.ECONOMIC, AgentType.NEWS_SENTIMENT]
                },
                timeout_seconds=240
            ),
            
            "market_analysis": AgentWorkflow(
                workflow_id="market_analysis",
                query_type="market_analysis",
                required_agents=[
                    AgentType.MARKET_INTELLIGENCE,
                    AgentType.NEWS_SENTIMENT,
                    AgentType.EDITORIAL_SYNTHESIS
                ],
                execution_order=[
                    [AgentType.NEWS_SENTIMENT],  # First: Get sentiment data
                    [AgentType.MARKET_INTELLIGENCE],  # Second: Market analysis with sentiment
                    [AgentType.EDITORIAL_SYNTHESIS]  # Final: Synthesis
                ],
                dependencies={
                    AgentType.MARKET_INTELLIGENCE: [AgentType.NEWS_SENTIMENT],
                    AgentType.EDITORIAL_SYNTHESIS: [AgentType.MARKET_INTELLIGENCE, AgentType.NEWS_SENTIMENT]
                },
                timeout_seconds=200
            )
        }
    
    def classify_query(self, query: str, context_hints: Optional[Dict] = None) -> str:
        """
        Classify the query type to determine appropriate workflow.
        
        Args:
            query: User query string
            context_hints: Optional hints about query context
            
        Returns:
            Query type string matching workflow keys
        """
        query_lower = query.lower()
        
        # Daily briefing patterns
        if any(phrase in query_lower for phrase in [
            "daily brief", "today's market", "market summary", "daily update",
            "what happened today", "market overview"
        ]):
            return "daily_briefing"
        
        # Deep dive patterns
        if any(phrase in query_lower for phrase in [
            "analyze", "deep dive", "detailed analysis", "explain why",
            "what's driving", "breakdown", "comprehensive"
        ]):
            return "deep_dive"
        
        # Correlation analysis patterns
        if any(phrase in query_lower for phrase in [
            "correlation", "relationship", "connection", "impact of",
            "how does", "affect", "influence"
        ]):
            return "correlation_analysis"
        
        # Market-specific analysis
        if any(phrase in query_lower for phrase in [
            "stock", "market", "sector", "technical", "chart",
            "price", "volume", "volatility"
        ]):
            return "market_analysis"
        
        # Default to deep dive for complex queries
        return "deep_dive"
    
    def assemble_context(
        self,
        query: str,
        query_type: str,
        timeframe: str = "1d",
        user_id: Optional[str] = None,
        additional_context: Optional[Dict] = None
    ) -> AgentContext:
        """
        Assemble comprehensive context for agent execution.
        
        Args:
            query: User query
            query_type: Classified query type
            timeframe: Analysis timeframe
            user_id: Optional user identifier
            additional_context: Additional context parameters
            
        Returns:
            AgentContext with all necessary information
        """
        # Determine date range based on timeframe
        end_date = datetime.now()
        if timeframe == "1d":
            start_date = end_date - timedelta(days=1)
        elif timeframe == "1w":
            start_date = end_date - timedelta(weeks=1)
        elif timeframe == "1m":
            start_date = end_date - timedelta(days=30)
        elif timeframe == "3m":
            start_date = end_date - timedelta(days=90)
        elif timeframe == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=1)  # Default to 1 day
        
        # Determine required data sources based on query type
        data_sources = []
        if query_type in ["daily_briefing", "deep_dive", "correlation_analysis"]:
            data_sources = ["fred", "yahoo_finance", "news_api"]
        elif query_type == "market_analysis":
            data_sources = ["yahoo_finance", "news_api"]
        
        context = AgentContext(
            query=query,
            query_type=query_type,
            timeframe=timeframe,
            user_id=user_id,
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            data_sources=data_sources,
            date_range={"start": start_date, "end": end_date},
            market_regime=self.shared_state.get("market_regime"),
            economic_cycle=self.shared_state.get("economic_cycle"),
            risk_environment=self.shared_state.get("risk_environment")
        )
        
        # Add any additional context
        if additional_context:
            for key, value in additional_context.items():
                if hasattr(context, key):
                    setattr(context, key, value)
        
        return context
    
    async def execute_workflow(
        self,
        workflow: AgentWorkflow,
        context: AgentContext
    ) -> SynthesizedResponse:
        """
        Execute a multi-agent workflow with proper coordination.
        
        Args:
            workflow: Workflow definition
            context: Execution context
            
        Returns:
            SynthesizedResponse with aggregated results
        """
        start_time = datetime.now()
        agent_responses: Dict[str, AgentResponse] = {}
        
        try:
            # Execute agents in defined order
            for execution_group in workflow.execution_order:
                # Filter to only include agents we have registered
                available_agents = [
                    agent_type for agent_type in execution_group
                    if agent_type in self.agent_registry
                ]
                
                if not available_agents:
                    logger.warning(f"No available agents for execution group: {execution_group}")
                    continue
                
                # Execute agents in parallel within each group
                tasks = []
                for agent_type in available_agents:
                    agent = self.agent_registry[agent_type]
                    task = asyncio.create_task(
                        agent.process_query(context),
                        name=f"{agent_type.value}_analysis"
                    )
                    tasks.append((agent_type, task))
                
                # Wait for all tasks in this group to complete
                group_results = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                )
                
                # Process results and update context
                for (agent_type, _), result in zip(tasks, group_results):
                    if isinstance(result, Exception):
                        logger.error(f"{agent_type.value} agent failed: {str(result)}")
                        continue
                    
                    agent_responses[agent_type.value] = result
                    context.add_agent_output(agent_type.value, result)
                    
                    # Update shared state from agent signals
                    if result.signals_for_other_agents:
                        self._update_shared_state(result.signals_for_other_agents)
            
            # Calculate overall confidence and execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            overall_confidence = self._calculate_overall_confidence(agent_responses)
            
            # Generate synthesis (this would typically be done by synthesis agent)
            synthesis_result = self._generate_synthesis(
                context, agent_responses, overall_confidence
            )
            
            return SynthesizedResponse(
                workflow_id=workflow.workflow_id,
                query=context.query,
                confidence=overall_confidence,
                executive_summary=synthesis_result["executive_summary"],
                key_insights=synthesis_result["key_insights"],
                cross_domain_signals=synthesis_result["cross_domain_signals"],
                risk_assessment=synthesis_result["risk_assessment"],
                agent_responses=agent_responses,
                synthesis_metadata=synthesis_result["metadata"],
                total_execution_time_ms=execution_time,
                agents_executed=list(agent_responses.keys())
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Workflow {workflow.workflow_id} timed out")
            raise
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
    
    def _update_shared_state(self, signals: Dict[str, Any]):
        """Update shared state based on agent signals"""
        if "market_regime" in signals:
            self.shared_state["market_regime"] = signals["market_regime"]
        if "economic_cycle" in signals:
            self.shared_state["economic_cycle"] = signals["economic_cycle"]
        if "risk_environment" in signals:
            self.shared_state["risk_environment"] = signals["risk_environment"]
        
        self.shared_state["last_updated"] = datetime.now()
    
    def _calculate_overall_confidence(
        self, 
        agent_responses: Dict[str, AgentResponse]
    ) -> float:
        """Calculate overall confidence from individual agent responses"""
        if not agent_responses:
            return 0.0
        
        # Weighted average based on agent importance and individual confidence
        weights = {
            "economic": 0.3,
            "market": 0.3,
            "sentiment": 0.25,
            "synthesis": 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent_type, response in agent_responses.items():
            weight = weights.get(agent_type, 0.1)
            weighted_sum += response.confidence * weight
            total_weight += weight
        
        return weighted_sum / max(total_weight, 1.0)
    
    def _generate_synthesis(
        self,
        context: AgentContext,
        agent_responses: Dict[str, AgentResponse],
        overall_confidence: float
    ) -> Dict[str, Any]:
        """
        Generate basic synthesis when synthesis agent is not available.
        This is a fallback - ideally synthesis agent would handle this.
        """
        # Collect all insights
        all_insights = []
        cross_domain_signals = []
        
        for agent_type, response in agent_responses.items():
            all_insights.extend(response.insights)
            
            # Look for cross-domain signals
            if response.signals_for_other_agents:
                cross_domain_signals.append({
                    "source_agent": agent_type,
                    "signals": response.signals_for_other_agents
                })
        
        # Generate executive summary
        executive_summary = f"Analysis of '{context.query}' across {len(agent_responses)} domains. "
        executive_summary += f"Overall confidence: {overall_confidence:.1%}. "
        
        if overall_confidence > 0.7:
            executive_summary += "High confidence in analysis with strong cross-domain alignment."
        elif overall_confidence > 0.5:
            executive_summary += "Moderate confidence with some uncertainty factors."
        else:
            executive_summary += "Lower confidence due to data limitations or conflicting signals."
        
        # Basic risk assessment
        risk_factors = []
        for response in agent_responses.values():
            risk_factors.extend(response.uncertainty_factors)
        
        risk_assessment = {
            "overall_risk": "medium" if len(risk_factors) > 2 else "low",
            "risk_factors": list(set(risk_factors)),
            "confidence_level": "high" if overall_confidence > 0.7 else "medium"
        }
        
        return {
            "executive_summary": executive_summary,
            "key_insights": all_insights[:10],  # Top 10 insights
            "cross_domain_signals": cross_domain_signals,
            "risk_assessment": risk_assessment,
            "metadata": {
                "synthesis_method": "orchestrator_fallback",
                "agents_contributing": list(agent_responses.keys()),
                "total_insights": len(all_insights)
            }
        }
    
    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Main analysis method for orchestration agent.
        Routes to appropriate workflow execution.
        """
        # Determine workflow
        workflow = self.workflows.get(context.query_type)
        if not workflow:
            raise ValueError(f"Unknown query type: {context.query_type}")
        
        # Execute workflow
        synthesis_result = await self.execute_workflow(workflow, context)
        
        # Convert to AgentResponse format
        return AgentResponse(
            agent_type=self.agent_type,
            confidence=synthesis_result.confidence,
            confidence_level=self._calculate_confidence_level(synthesis_result.confidence),
            analysis={
                "executive_summary": synthesis_result.executive_summary,
                "workflow_executed": workflow.workflow_id,
                "agents_coordinated": synthesis_result.agents_executed
            },
            insights=synthesis_result.key_insights,
            key_metrics={
                "total_execution_time_ms": synthesis_result.total_execution_time_ms,
                "agents_executed": len(synthesis_result.agents_executed),
                "cross_domain_signals": len(synthesis_result.cross_domain_signals)
            },
            data_sources_used=context.data_sources,
            timeframe_analyzed=context.timeframe,
            execution_time_ms=synthesis_result.total_execution_time_ms,
            signals_for_other_agents={
                "synthesis_complete": True,
                "overall_confidence": synthesis_result.confidence
            }
        )
    
    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return data sources required for orchestration"""
        # Orchestrator needs access to all data sources for coordination
        return ["fred", "yahoo_finance", "news_api", "chroma_db"]
    
    async def process_user_query(
        self,
        query: str,
        timeframe: str = "1d",
        user_id: Optional[str] = None,
        additional_context: Optional[Dict] = None
    ) -> SynthesizedResponse:
        """
        Main entry point for processing user queries.
        
        Args:
            query: User query string
            timeframe: Analysis timeframe
            user_id: Optional user identifier
            additional_context: Additional context parameters
            
        Returns:
            SynthesizedResponse with complete analysis
        """
        # Classify query and assemble context
        query_type = self.classify_query(query, additional_context)
        context = self.assemble_context(
            query, query_type, timeframe, user_id, additional_context
        )
        
        # Get appropriate workflow
        workflow = self.workflows.get(query_type)
        if not workflow:
            raise ValueError(f"No workflow available for query type: {query_type}")
        
        # Execute workflow
        logger.info(f"Executing {query_type} workflow for query: '{query}'")
        return await self.execute_workflow(workflow, context)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of all available workflows"""
        return {
            "available_workflows": list(self.workflows.keys()),
            "registered_agents": [agent_type.value for agent_type in self.agent_registry.keys()],
            "shared_state": self.shared_state,
            "performance_stats": self.get_performance_stats()
        }
