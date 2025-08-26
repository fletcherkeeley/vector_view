#!/usr/bin/env python3
"""
Test embedding pipeline functionality
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from semantic.embedding_pipeline import create_embedding_pipeline

async def run_embedding():
    """Test embedding pipeline"""
    print('🔗 Starting embedding pipeline...')
    
    try:
        pipeline = await create_embedding_pipeline()
        results = await pipeline.run_full_embedding_pipeline()
        
        news_processed = results.get('news_articles', {}).get('processed', 0)
        indicators_processed = results.get('economic_indicators', {}).get('processed', 0)
        
        print(f'✅ Embedding pipeline completed:')
        print(f'📰 News articles processed: {news_processed}')
        print(f'📊 Economic indicators processed: {indicators_processed}')
        
        return results
        
    except Exception as e:
        print(f'❌ Embedding pipeline failed: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(run_embedding())
