#!/usr/bin/env python3
"""
Quick test to verify AI service fixes for <think> tag removal
"""

import asyncio
import logging
from agents.ai_service import OllamaService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ai_service_fix():
    """Test AI service with think tag removal"""
    
    print("🧪 Testing AI Service Fix for <think> tags...")
    
    # Initialize AI service
    ai_service = OllamaService(model="qwen3:32b")
    
    # Test health check
    health = await ai_service.health_check()
    print(f"📡 Ollama Health: {'✅ Connected' if health else '❌ Disconnected'}")
    
    if not health:
        print("❌ Cannot test - Ollama not available")
        return
    
    # Test economic analysis with mock data
    print("\n📊 Testing Economic Analysis...")
    
    mock_indicators = {
        "UNRATE": {"current_value": 3.7, "change_percent": -0.1},
        "FEDFUNDS": {"current_value": 5.25, "change_percent": 0.0},
        "CPIAUCSL": {"current_value": 307.2, "change_percent": 0.2}
    }
    
    mock_trends = {
        "UNRATE": {"trend_direction": "stable", "current_value": 3.7},
        "FEDFUNDS": {"trend_direction": "stable", "current_value": 5.25}
    }
    
    mock_correlations = {
        "strong_correlations": [
            {"indicator_1": "UNRATE", "indicator_2": "FEDFUNDS", "correlation": -0.65}
        ]
    }
    
    try:
        result = await ai_service.analyze_economic_data(
            indicators_data=mock_indicators,
            trends=mock_trends,
            correlations=mock_correlations,
            context="Test analysis for unemployment and Fed policy"
        )
        
        print(f"✅ Analysis completed")
        print(f"📈 Confidence: {result.confidence:.3f}")
        print(f"📝 Content length: {len(result.content)} chars")
        print(f"🔍 Key points: {len(result.key_points)}")
        
        # Check if <think> tags were removed
        has_think_tags = "<think>" in result.content
        print(f"🧠 Think tags removed: {'❌ Still present' if has_think_tags else '✅ Cleaned'}")
        
        # Show first 200 chars of content
        print(f"\n📄 Content preview:")
        print(f"   {result.content[:200]}{'...' if len(result.content) > 200 else ''}")
        
        if result.key_points:
            print(f"\n💡 Key points:")
            for i, point in enumerate(result.key_points[:3], 1):
                print(f"   {i}. {point[:100]}{'...' if len(point) > 100 else ''}")
        
    except Exception as e:
        print(f"❌ Economic analysis failed: {str(e)}")
    
    print(f"\n📊 AI Service Stats:")
    stats = ai_service.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_ai_service_fix())
