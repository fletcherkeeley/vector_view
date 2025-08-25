"""
AI Service Layer for Vector View Agents

Provides LLM integration via Ollama API for intelligent analysis and synthesis.
Handles prompt engineering, response parsing, and error handling for all agents.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    """Structured response from AI service"""
    content: str
    confidence: float
    reasoning: List[str]
    key_points: List[str]
    uncertainty_factors: List[str]
    metadata: Dict[str, Any]


class OllamaService:
    """
    Service for interacting with Ollama API for AI-powered analysis.
    
    Provides specialized methods for different types of financial analysis:
    - Economic indicator interpretation
    - Market trend analysis  
    - News sentiment analysis
    - Cross-domain synthesis
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        timeout: int = 120,  # Longer timeout for larger model
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.avg_response_time = 0.0
        
        logger.info(f"Initialized Ollama service: {base_url} with model {model}")
    
    async def _make_request(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Make request to Ollama API with retry logic"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for more consistent analysis
                "top_p": 0.9,
                "num_predict": 1000
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = datetime.now()
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Update performance metrics
                            response_time = (datetime.now() - start_time).total_seconds()
                            self.request_count += 1
                            self.avg_response_time = (
                                (self.avg_response_time * (self.request_count - 1) + response_time) 
                                / self.request_count
                            )
                            
                            return result
                        else:
                            error_text = await response.text()
                            raise Exception(f"Ollama API error {response.status}: {error_text}")
                            
            except Exception as e:
                last_exception = e
                logger.warning(f"Ollama request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"All {self.max_retries} Ollama requests failed. Last error: {str(last_exception)}")
    
    async def analyze_economic_data(
        self,
        indicators_data: Dict[str, Any],
        trends: Dict[str, Any],
        correlations: Dict[str, Any],
        context: str = ""
    ) -> AIResponse:
        """
        AI-powered analysis of economic indicators and trends.
        
        Args:
            indicators_data: Raw economic indicator values
            trends: Statistical trend analysis
            correlations: Correlation analysis results
            context: Additional context about the query
            
        Returns:
            AIResponse with intelligent economic analysis
        """
        
        system_prompt = """You are an expert economic analyst with deep knowledge of macroeconomic indicators, Federal Reserve policy, and economic cycles. 

Your role is to:
1. Interpret economic data patterns and trends
2. Assess economic cycle positioning 
3. Identify policy implications and market impacts
4. Provide actionable insights for investors and analysts
5. Quantify confidence levels and uncertainty factors

Always provide specific, data-driven insights with clear reasoning."""

        # Format the data for the prompt
        data_summary = self._format_economic_data(indicators_data, trends, correlations)
        
        prompt = f"""Analyze the following economic data and provide intelligent insights:

ECONOMIC INDICATORS DATA:
{data_summary}

ANALYSIS CONTEXT:
{context}

Please provide a comprehensive analysis including:

1. ECONOMIC ASSESSMENT: What do these indicators tell us about the current economic state?

2. KEY INSIGHTS: What are the 3-5 most important takeaways from this data?

3. POLICY IMPLICATIONS: What do these trends suggest about monetary/fiscal policy?

4. MARKET IMPLICATIONS: How should investors interpret these signals?

5. CONFIDENCE & UNCERTAINTY: Rate your confidence (0-100%) and identify key uncertainty factors.

6. FORWARD OUTLOOK: What do these indicators suggest about near-term economic direction?

Format your response as structured analysis with clear reasoning for each point."""

        try:
            result = await self._make_request(prompt, system_prompt)
            response_text = result.get('response', '')
            
            # Parse the AI response into structured format
            return self._parse_economic_response(response_text)
            
        except Exception as e:
            logger.error(f"Economic analysis failed: {str(e)}")
            return AIResponse(
                content=f"Economic analysis unavailable: {str(e)}",
                confidence=0.0,
                reasoning=["AI analysis failed"],
                key_points=[],
                uncertainty_factors=["ai_service_error"],
                metadata={"error": str(e)}
            )
    
    async def analyze_market_data(
        self,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        context: str = ""
    ) -> AIResponse:
        """AI-powered market analysis"""
        
        system_prompt = """You are an expert market analyst specializing in technical analysis, sector rotation, and market psychology.

Your expertise includes:
- Technical pattern recognition and trend analysis
- Sector and asset class performance interpretation  
- Market volatility and risk assessment
- Cross-asset correlations and portfolio implications
- Market regime identification (bull/bear/sideways)

Provide actionable market insights with quantified confidence levels."""

        data_summary = self._format_market_data(market_data, technical_indicators)
        
        prompt = f"""Analyze the following market data and provide intelligent insights:

MARKET DATA:
{data_summary}

ANALYSIS CONTEXT:
{context}

Please provide comprehensive market analysis including:

1. MARKET REGIME: What market environment are we in? (bull/bear/sideways/volatile)

2. TECHNICAL SIGNALS: What do the price patterns and indicators suggest?

3. SECTOR INSIGHTS: Which sectors/assets are showing strength or weakness?

4. RISK ASSESSMENT: What are the key risk factors and volatility patterns?

5. TRADING IMPLICATIONS: What should traders and investors focus on?

6. CONFIDENCE & UNCERTAINTY: Rate confidence (0-100%) and key uncertainty factors.

Provide specific, actionable insights with clear reasoning."""

        try:
            result = await self._make_request(prompt, system_prompt)
            response_text = result.get('response', '')
            return self._parse_market_response(response_text)
            
        except Exception as e:
            logger.error(f"Market analysis failed: {str(e)}")
            return AIResponse(
                content=f"Market analysis unavailable: {str(e)}",
                confidence=0.0,
                reasoning=["AI analysis failed"],
                key_points=[],
                uncertainty_factors=["ai_service_error"],
                metadata={"error": str(e)}
            )
    
    async def analyze_news_sentiment(
        self,
        news_articles: List[Dict[str, Any]],
        semantic_context: Dict[str, Any],
        context: str = ""
    ) -> AIResponse:
        """AI-powered news sentiment and narrative analysis"""
        
        system_prompt = """You are an expert financial news analyst specializing in market sentiment, narrative analysis, and news impact assessment.

Your expertise includes:
- Financial news sentiment analysis and market psychology
- Central bank communication interpretation
- Geopolitical event impact assessment
- Corporate earnings and guidance analysis
- News flow narrative tracking and theme identification

Provide nuanced sentiment analysis with market impact implications."""

        news_summary = self._format_news_data(news_articles, semantic_context)
        
        prompt = f"""Analyze the following news data for sentiment and market implications:

NEWS DATA:
{news_summary}

ANALYSIS CONTEXT:
{context}

Please provide comprehensive sentiment analysis including:

1. OVERALL SENTIMENT: What's the dominant market sentiment from this news flow?

2. KEY NARRATIVES: What are the main themes and stories driving sentiment?

3. MARKET IMPACT: How is this news likely to affect different markets/sectors?

4. SENTIMENT DRIVERS: What specific events or statements are most impactful?

5. FORWARD IMPLICATIONS: How might this sentiment evolve?

6. CONFIDENCE & UNCERTAINTY: Rate confidence (0-100%) and uncertainty factors.

Focus on actionable sentiment insights with clear market implications."""

        try:
            result = await self._make_request(prompt, system_prompt)
            response_text = result.get('response', '')
            return self._parse_sentiment_response(response_text)
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return AIResponse(
                content=f"Sentiment analysis unavailable: {str(e)}",
                confidence=0.0,
                reasoning=["AI analysis failed"],
                key_points=[],
                uncertainty_factors=["ai_service_error"],
                metadata={"error": str(e)}
            )
    
    async def synthesize_analysis(
        self,
        economic_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        query: str,
        context: str = ""
    ) -> AIResponse:
        """AI-powered synthesis across all analysis domains"""
        
        system_prompt = """You are a senior financial strategist responsible for synthesizing multi-domain analysis into coherent investment intelligence.

Your role is to:
- Integrate economic, market, and sentiment analysis into unified insights
- Identify cross-domain patterns and contradictions
- Generate actionable investment recommendations
- Assess overall market environment and positioning
- Provide executive-level strategic guidance

Create comprehensive, actionable intelligence for investment decision-making."""

        synthesis_data = self._format_synthesis_data(economic_analysis, market_analysis, sentiment_analysis)
        
        prompt = f"""Synthesize the following multi-domain analysis into unified investment intelligence:

QUERY: {query}

MULTI-DOMAIN ANALYSIS:
{synthesis_data}

CONTEXT: {context}

Please provide comprehensive synthesis including:

1. EXECUTIVE SUMMARY: What's the unified picture across all domains?

2. KEY INSIGHTS: What are the most important cross-domain insights?

3. INVESTMENT IMPLICATIONS: What should investors do based on this analysis?

4. RISK ASSESSMENT: What are the primary risks and opportunities?

5. MARKET POSITIONING: How should portfolios be positioned?

6. CONFIDENCE & UNCERTAINTY: Overall confidence (0-100%) and key uncertainties.

7. ACTION ITEMS: Specific, actionable recommendations.

Provide clear, executive-level strategic guidance with supporting reasoning."""

        try:
            result = await self._make_request(prompt, system_prompt)
            response_text = result.get('response', '')
            return self._parse_synthesis_response(response_text)
            
        except Exception as e:
            logger.error(f"Synthesis analysis failed: {str(e)}")
            return AIResponse(
                content=f"Synthesis analysis unavailable: {str(e)}",
                confidence=0.0,
                reasoning=["AI synthesis failed"],
                key_points=[],
                uncertainty_factors=["ai_service_error"],
                metadata={"error": str(e)}
            )
    
    def _format_economic_data(self, indicators_data: Dict, trends: Dict, correlations: Dict) -> str:
        """Format economic data for AI prompt"""
        formatted = []
        
        # Format indicator trends
        if trends:
            formatted.append("INDICATOR TRENDS:")
            for indicator, trend_data in trends.items():
                formatted.append(f"- {indicator}: {trend_data.get('current_value', 'N/A')} "
                                f"({trend_data.get('change_percent', 0):+.1f}%, "
                                f"trend: {trend_data.get('trend_direction', 'unknown')})")
        
        # Format correlations
        if correlations.get('strong_correlations'):
            formatted.append("\nSTRONG CORRELATIONS:")
            for corr in correlations['strong_correlations'][:3]:  # Top 3
                formatted.append(f"- {corr['indicator_1']} ↔ {corr['indicator_2']}: "
                               f"{corr['correlation']:.2f}")
        
        return "\n".join(formatted) if formatted else "No economic data available"
    
    def _format_market_data(self, market_data: Dict, technical_indicators: Dict) -> str:
        """Format market data for AI prompt"""
        formatted = []
        
        if market_data:
            formatted.append("MARKET DATA:")
            for asset, data in market_data.items():
                formatted.append(f"- {asset}: {data}")
        
        if technical_indicators:
            formatted.append("\nTECHNICAL INDICATORS:")
            for indicator, value in technical_indicators.items():
                formatted.append(f"- {indicator}: {value}")
        
        return "\n".join(formatted) if formatted else "No market data available"
    
    def _format_news_data(self, news_articles: List[Dict], semantic_context: Dict) -> str:
        """Format news data for AI prompt"""
        formatted = []
        
        if news_articles:
            formatted.append("NEWS ARTICLES:")
            for i, article in enumerate(news_articles[:5], 1):  # Top 5 articles
                formatted.append(f"{i}. {article.get('title', 'No title')}")
                if article.get('description'):
                    formatted.append(f"   {article['description'][:200]}...")
        
        if semantic_context:
            formatted.append(f"\nSEMANTIC CONTEXT: {semantic_context}")
        
        return "\n".join(formatted) if formatted else "No news data available"
    
    def _format_synthesis_data(self, economic: Dict, market: Dict, sentiment: Dict) -> str:
        """Format multi-domain data for synthesis"""
        formatted = []
        
        if economic:
            formatted.append(f"ECONOMIC ANALYSIS:\n{economic.get('summary', 'No economic analysis')}")
        
        if market:
            formatted.append(f"\nMARKET ANALYSIS:\n{market.get('summary', 'No market analysis')}")
        
        if sentiment:
            formatted.append(f"\nSENTIMENT ANALYSIS:\n{sentiment.get('summary', 'No sentiment analysis')}")
        
        return "\n".join(formatted) if formatted else "No analysis data available"
    
    def _parse_economic_response(self, response_text: str) -> AIResponse:
        """Parse AI response for economic analysis"""
        return self._parse_generic_response(response_text, "economic")
    
    def _parse_market_response(self, response_text: str) -> AIResponse:
        """Parse AI response for market analysis"""
        return self._parse_generic_response(response_text, "market")
    
    def _parse_sentiment_response(self, response_text: str) -> AIResponse:
        """Parse AI response for sentiment analysis"""
        return self._parse_generic_response(response_text, "sentiment")
    
    def _parse_synthesis_response(self, response_text: str) -> AIResponse:
        """Parse AI response for synthesis"""
        return self._parse_generic_response(response_text, "synthesis")
    
    def _parse_generic_response(self, response_text: str, analysis_type: str) -> AIResponse:
        """Parse generic AI response and extract structured information"""
        try:
            # Clean response text - remove <think> tags and content (both encoded and plain)
            import re
            cleaned_text = re.sub(r'&lt;think&gt;.*?&lt;/think&gt;', '', response_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'# <think>.*', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'^<think>.*', '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
            cleaned_text = cleaned_text.strip()
            
            # If response is empty after cleaning, use original but warn
            if not cleaned_text:
                cleaned_text = response_text
                logger.warning(f"AI response contained only thinking tags for {analysis_type}")
        except Exception as e:
            logger.warning(f"Error cleaning AI response: {str(e)}")
            cleaned_text = response_text
        
        # Extract confidence if mentioned
        confidence = 0.7  # Default confidence
        confidence_keywords = ["confidence", "certain", "sure"]
        
        # Look for percentage mentions
        confidence_matches = re.findall(r'(\d+)%', cleaned_text.lower())
        if confidence_matches:
            try:
                confidence = float(confidence_matches[-1]) / 100  # Use last percentage found
            except:
                pass
        
        # Extract key points (look for numbered lists or bullet points)
        key_points = []
        lines = cleaned_text.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')) and 
                len(line) > 10):  # Reasonable length
                key_points.append(line)
        
        # Extract uncertainty factors
        uncertainty_factors = []
        uncertainty_keywords = ["uncertain", "unclear", "risk", "volatility", "unknown"]
        for keyword in uncertainty_keywords:
            if keyword in cleaned_text.lower():
                uncertainty_factors.append(keyword)
        
        # Extract reasoning (look for explanatory text)
        reasoning = []
        for line in lines:
            if any(word in line.lower() for word in ["because", "due to", "given", "since"]):
                reasoning.append(line.strip())
        
        return AIResponse(
            content=cleaned_text,
            confidence=min(max(confidence, 0.0), 1.0),  # Clamp between 0 and 1
            reasoning=reasoning[:5],  # Top 5 reasoning points
            key_points=key_points[:10],  # Top 10 key points
            uncertainty_factors=list(set(uncertainty_factors)),  # Remove duplicates
            metadata={
                "analysis_type": analysis_type,
                "response_length": len(response_text),
                "model_used": self.model
            }
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get AI service performance statistics"""
        return {
            "request_count": self.request_count,
            "avg_response_time": self.avg_response_time,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "base_url": self.base_url
        }
    
    async def generate_response(self, prompt: str, context: str = "", max_tokens: int = 1000) -> str:
        """Generate a simple text response for general AI tasks"""
        try:
            system_prompt = f"You are a financial intelligence AI assistant. Context: {context}"
            response = await self._make_request(prompt, system_prompt)
            return response.get('response', 'No response generated')
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return f"Unable to generate response: {str(e)}"

    async def health_check(self) -> bool:
        """Check if Ollama service is available"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except:
            return False
