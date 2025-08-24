"""
Base Agent Class for Vector View Financial Intelligence Platform

Provides the foundational architecture for all AI agents in the system.
Each agent specializes in a domain (economic, market, sentiment, synthesis)
while maintaining consistent interfaces and communication protocols.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of agent types in the system"""
    ORCHESTRATION = "orchestration"
    ECONOMIC = "economic"
    MARKET = "market"
    MARKET_INTELLIGENCE = "market_intelligence"
    SENTIMENT = "sentiment"
    NEWS_SENTIMENT = "news_sentiment"
    SYNTHESIS = "synthesis"
    EDITORIAL_SYNTHESIS = "editorial_synthesis"
    GEOPOLITICAL = "geopolitical"
    SECTOR_ANALYSIS = "sector_analysis"
    BREAKING_NEWS = "breaking_news"
    RESEARCH = "research"


class ConfidenceLevel(Enum):
    """Confidence levels for agent responses"""
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.79
    LOW = "low"        # 0.0-0.49


@dataclass
class AgentContext:
    """
    Context object passed between agents containing query information,
    timeframes, data sources, and cross-agent state.
    """
    query: str
    query_type: str  # "daily_briefing", "deep_dive", "correlation_analysis"
    timeframe: str   # "1d", "1w", "1m", "3m", "1y"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Data context
    data_sources: List[str] = field(default_factory=list)
    date_range: Optional[Dict[str, datetime]] = None
    
    # Cross-agent shared state
    market_regime: Optional[str] = None  # "bull", "bear", "sideways", "volatile"
    economic_cycle: Optional[str] = None  # "expansion", "peak", "contraction", "trough"
    risk_environment: Optional[str] = None  # "risk_on", "risk_off", "neutral"
    
    # Agent execution tracking
    agents_executed: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    
    def add_agent_output(self, agent_type: str, output: Any):
        """Add output from an agent to shared context"""
        self.agent_outputs[agent_type] = output
        if agent_type not in self.agents_executed:
            self.agents_executed.append(agent_type)


@dataclass
class AgentResponse:
    """
    Standardized response format for all agents
    """
    agent_type: AgentType
    confidence: float  # 0.0 to 1.0
    confidence_level: ConfidenceLevel
    
    # Core analysis results
    analysis: Dict[str, Any]
    insights: List[str]
    key_metrics: Dict[str, float]
    
    # Supporting information
    data_sources_used: List[str]
    timeframe_analyzed: str
    execution_time_ms: float
    
    # Cross-agent communication
    signals_for_other_agents: Dict[str, Any] = field(default_factory=dict)
    uncertainty_factors: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization"""
        return {
            "agent_type": self.agent_type.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "analysis": self.analysis,
            "insights": self.insights,
            "key_metrics": self.key_metrics,
            "data_sources_used": self.data_sources_used,
            "timeframe_analyzed": self.timeframe_analyzed,
            "execution_time_ms": self.execution_time_ms,
            "signals_for_other_agents": self.signals_for_other_agents,
            "uncertainty_factors": self.uncertainty_factors,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version
        }


class BaseAgent(ABC):
    """
    Abstract base class for all Vector View AI agents.
    
    Provides common functionality for database access, logging, caching,
    and standardized response formatting.
    """
    
    def __init__(
        self, 
        agent_type: AgentType,
        database_url: str,
        cache_ttl_minutes: int = 30,
        max_retries: int = 3
    ):
        self.agent_type = agent_type
        self.database_url = database_url
        self.cache_ttl_minutes = cache_ttl_minutes
        self.max_retries = max_retries
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{agent_type.value}")
        
        # Agent memory and caching
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Performance tracking
        self._execution_stats = {
            "total_queries": 0,
            "avg_execution_time_ms": 0.0,
            "cache_hit_rate": 0.0
        }
        
        self.logger.info(f"Initialized {agent_type.value} agent")
    
    @abstractmethod
    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Main analysis method that each agent must implement.
        
        Args:
            context: AgentContext containing query and shared state
            
        Returns:
            AgentResponse with analysis results and insights
        """
        pass
    
    @abstractmethod
    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """
        Return list of data sources required for analysis.
        
        Args:
            context: AgentContext containing query information
            
        Returns:
            List of required data source identifiers
        """
        pass
    
    def _calculate_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level enum"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _get_cache_key(self, context: AgentContext) -> str:
        """Generate cache key for context"""
        key_components = [
            context.query_type,
            context.timeframe,
            str(context.date_range) if context.date_range else "no_date",
            "_".join(sorted(context.data_sources))
        ]
        return f"{self.agent_type.value}:{':'.join(key_components)}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_age = datetime.now() - self._cache_timestamps[cache_key]
        return cache_age < timedelta(minutes=self.cache_ttl_minutes)
    
    def _cache_result(self, cache_key: str, result: AgentResponse):
        """Cache analysis result"""
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()
    
    def _get_cached_result(self, cache_key: str) -> Optional[AgentResponse]:
        """Retrieve cached result if valid"""
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"Cache hit for key: {cache_key}")
            return self._cache[cache_key]
        return None
    
    async def _execute_with_retry(self, context: AgentContext) -> AgentResponse:
        """Execute analysis with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = datetime.now()
                result = await self.analyze(context)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Update execution stats
                self._execution_stats["total_queries"] += 1
                current_avg = self._execution_stats["avg_execution_time_ms"]
                total_queries = self._execution_stats["total_queries"]
                self._execution_stats["avg_execution_time_ms"] = (
                    (current_avg * (total_queries - 1) + execution_time) / total_queries
                )
                
                result.execution_time_ms = execution_time
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        self.logger.error(f"All {self.max_retries} attempts failed. Last error: {str(last_exception)}")
        raise last_exception
    
    async def process_query(self, context: AgentContext) -> AgentResponse:
        """
        Main entry point for agent query processing.
        Handles caching, retries, and performance tracking.
        """
        # Check cache first
        cache_key = self._get_cache_key(context)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Execute analysis with retry logic
        try:
            result = await self._execute_with_retry(context)
            
            # Cache successful result
            self._cache_result(cache_key, result)
            
            self.logger.info(
                f"Analysis completed - Confidence: {result.confidence:.2f}, "
                f"Execution time: {result.execution_time_ms:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent analysis failed: {str(e)}")
            # Return error response
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                analysis={"error": str(e)},
                insights=[f"Analysis failed due to: {str(e)}"],
                key_metrics={},
                data_sources_used=[],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=0.0,
                uncertainty_factors=["analysis_failure"]
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return agent performance statistics"""
        cache_hits = sum(1 for key in self._cache_timestamps.keys() 
                        if self._is_cache_valid(key))
        total_cache_entries = len(self._cache_timestamps)
        
        self._execution_stats["cache_hit_rate"] = (
            cache_hits / max(total_cache_entries, 1)
        )
        
        return {
            **self._execution_stats,
            "cache_entries": total_cache_entries,
            "cache_hits": cache_hits
        }
    
    def clear_cache(self):
        """Clear agent cache"""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Agent cache cleared")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type.value})"
