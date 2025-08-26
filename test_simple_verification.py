#!/usr/bin/env python3
"""
Simple verification of news articles and ChromaDB storage
"""
import asyncio
import os
import sys
from pathlib import Path
import chromadb

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def verify_storage():
    """Simple verification of storage and embeddings"""
    
    print("🔍 Verifying news article storage and embeddings...")
    
    # Check ChromaDB embeddings
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Check news collection
        news_collection = chroma_client.get_collection("news_articles")
        news_count = news_collection.count()
        
        # Check economic indicators collection  
        indicators_collection = chroma_client.get_collection("economic_indicators")
        indicators_count = indicators_collection.count()
        
        print(f"🔗 ChromaDB Embeddings:")
        print(f"  News articles in ChromaDB: {news_count}")
        print(f"  Economic indicators in ChromaDB: {indicators_count}")
        
        # Test semantic search on recent articles
        search_results = news_collection.query(
            query_texts=["Federal Reserve interest rates policy"],
            n_results=5
        )
        
        print(f"\n🔍 Sample Semantic Search (Federal Reserve):")
        for i, (doc_id, distance) in enumerate(zip(search_results['ids'][0], search_results['distances'][0])):
            metadata = search_results['metadatas'][0][i]
            title = metadata.get('title', 'No title')[:60] + "..."
            similarity = 1 - distance
            print(f"  {i+1}. {title} (similarity: {similarity:.3f})")
        
        # Check for recent articles (today's embeddings)
        recent_search = news_collection.query(
            query_texts=["employment inflation economic"],
            n_results=10,
            where={"created_date": {"$gte": "2025-08-26"}}
        )
        
        recent_count = len(recent_search['ids'][0]) if recent_search['ids'] else 0
        print(f"\n📅 Recent articles (today): {recent_count} found in ChromaDB")
        
        return {
            'news_articles': news_count,
            'economic_indicators': indicators_count,
            'recent_articles': recent_count,
            'search_working': len(search_results['ids'][0]) > 0
        }
            
    except Exception as e:
        print(f"❌ ChromaDB verification failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(verify_storage())
