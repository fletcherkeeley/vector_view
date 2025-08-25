"""
Test script for AI-powered Vector View Agents

Tests the AI integration with Ollama for intelligent analysis
and validates that agents provide AI-generated insights.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestration_agent import OrchestrationAgent
from agents.economic import EconomicAnalysisAgent
from agents.ai_service import OllamaService
from agents.base_agent import AgentContext

# Load environment variables
load_dotenv()

async def test_ollama_service():
    """Test basic Ollama service connectivity"""
    print("🤖 Testing Ollama Service...")
    
    ai_service = OllamaService()
    
    # Health check
    is_healthy = await ai_service.health_check()
    print(f"  🔍 Ollama health check: {'✅ Connected' if is_healthy else '❌ Failed'}")
    
    if not is_healthy:
        print("  ⚠️  Make sure Ollama is running: `ollama serve`")
        print("  ⚠️  And model is available: `ollama pull qwen3:32b`")
        return False
    
    # Test basic AI analysis
    try:
        print("  🧠 Testing AI economic analysis...")
        
        test_data = {
            "UNRATE": {"current_value": 3.7, "trend_direction": "stable", "change_percent": 0.1},
            "FEDFUNDS": {"current_value": 5.25, "trend_direction": "stable", "change_percent": 0.0}
        }
        
        ai_response = await ai_service.analyze_economic_data(
            indicators_data={"UNRATE": [3.7], "FEDFUNDS": [5.25]},
            trends=test_data,
            correlations={"strong_correlations": []},
            context="Test analysis of unemployment and federal funds rate"
        )
        
        print(f"  ✅ AI Analysis completed!")
        print(f"  📊 Confidence: {ai_response.confidence:.1%}")
        print(f"  💡 Key points: {len(ai_response.key_points)}")
        print(f"  🔍 First insight: {ai_response.key_points[0] if ai_response.key_points else 'No insights'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ AI analysis failed: {str(e)}")
        return False

async def test_ai_economic_agent():
    """Test AI-powered Economic Analysis Agent"""
    print("\n🧪 Testing AI-Powered Economic Agent...")
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/vector_view')
    
    # Initialize AI-powered agent
    economic_agent = EconomicAnalysisAgent(
        database_url=database_url,
        ollama_url="http://localhost:11434",
        ollama_model="qwen3:32b"
    )
    
    # Create test context with realistic date range
    context = AgentContext(
        query="What do recent economic indicators tell us about inflation and employment trends?",
        query_type="economic_analysis",
        timeframe="3m",
        data_sources=["fred"],
        date_range={
            "start": datetime(2025, 4, 1),
            "end": datetime(2025, 7, 31)
        }
    )
    
    try:
        print("  🧠 Running AI-powered economic analysis...")
        response = await economic_agent.process_query(context)
        
        print(f"  ✅ AI Analysis completed!")
        print(f"  📈 Confidence: {response.confidence:.1%} ({response.confidence_level.value})")
        print(f"  ⏱️  Execution time: {response.execution_time_ms:.1f}ms")
        print(f"  📋 Indicators analyzed: {response.key_metrics.get('indicators_analyzed', 0)}")
        
        print(f"\n  🧠 AI-Generated Insights:")
        for i, insight in enumerate(response.insights[:3], 1):
            print(f"    {i}. {insight}")
        
        # Check if we have AI analysis content
        if "ai_analysis" in response.analysis:
            ai_content = response.analysis["ai_analysis"]
            print(f"\n  📝 AI Analysis Preview:")
            print(f"    {ai_content[:200]}..." if len(ai_content) > 200 else f"    {ai_content}")
        
        print(f"\n  🔗 Cross-agent signals: {len(response.signals_for_other_agents)} signals")
        for signal, value in response.signals_for_other_agents.items():
            print(f"    • {signal}: {value}")
        
        return response
        
    except Exception as e:
        print(f"  ❌ AI Economic agent test failed: {str(e)}")
        return None

async def test_ai_orchestration():
    """Test AI-powered orchestration workflow"""
    print("\n🎭 Testing AI-Powered Orchestration...")
    
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/vector_view')
    
    # Initialize orchestrator and AI-powered agents
    orchestrator = OrchestrationAgent(database_url=database_url)
    economic_agent = EconomicAnalysisAgent(
        database_url=database_url,
        ollama_url="http://localhost:11434",
        ollama_model="qwen3:32b"
    )
    
    # Register AI-powered agent
    orchestrator.register_agent(economic_agent)
    
    try:
        print("  🚀 Testing AI-powered workflow execution...")
        response = await orchestrator.process_user_query(
            query="Provide an intelligent analysis of current US economic conditions based on key indicators",
            timeframe="3m"
        )
        
        print(f"  ✅ AI Workflow completed!")
        print(f"  📈 Overall confidence: {response.confidence:.1%}")
        print(f"  ⏱️  Total execution time: {response.total_execution_time_ms:.1f}ms")
        print(f"  🤖 Agents executed: {', '.join(response.agents_executed)}")
        
        print(f"\n  📋 Executive Summary:")
        print(f"    {response.executive_summary}")
        
        print(f"\n  🧠 AI-Generated Insights:")
        for i, insight in enumerate(response.key_insights[:3], 1):
            print(f"    {i}. {insight}")
        
        return response
        
    except Exception as e:
        print(f"  ❌ AI Orchestration test failed: {str(e)}")
        return None

async def main():
    """Run all AI agent tests"""
    print("🤖 Vector View AI Agent Testing Suite")
    print("=" * 60)
    
    # Test Ollama service first
    ollama_working = await test_ollama_service()
    
    if not ollama_working:
        print("\n❌ Ollama service not available. Please ensure:")
        print("   1. Ollama is installed and running: `ollama serve`")
        print("   2. Model is available: `ollama pull llama3.1:8b`")
        return
    
    # Test AI-powered economic agent
    economic_response = await test_ai_economic_agent()
    
    # Test AI-powered orchestration
    orchestration_response = await test_ai_orchestration()
    
    print("\n" + "=" * 60)
    if economic_response and orchestration_response:
        print("✅ All AI agent tests completed successfully!")
        print("🧠 Agents are now using AI for intelligent analysis")
        print(f"📊 Economic agent confidence: {economic_response.confidence:.1%}")
        print(f"🎭 Orchestration confidence: {orchestration_response.confidence:.1%}")
    else:
        print("❌ Some AI tests failed - check Ollama setup and error messages above")

if __name__ == "__main__":
    asyncio.run(main())
