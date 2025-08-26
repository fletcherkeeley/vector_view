#!/usr/bin/env python3
"""
Verify articles are properly stored and embedded in ChromaDB
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from database.unified_database_setup import DatabaseManager
from sqlalchemy import select, func, text
import chromadb

async def verify_storage_and_embeddings():
    """Verify articles are stored in PostgreSQL and embedded in ChromaDB"""
    
    print("🔍 Verifying news article storage and embeddings...")
    
    # Initialize database manager
    database_url = os.getenv('DATABASE_URL')
    db_manager = DatabaseManager(database_url)
    await db_manager.initialize_engine()
    
    # Check PostgreSQL storage
    async with db_manager.AsyncSessionLocal() as session:
        # Count total articles
        total_articles = await session.execute(text("SELECT COUNT(*) FROM news_articles"))
        total_count = total_articles.scalar()
        
        # Count articles with embeddings
        embedded_articles = await session.execute(text("SELECT COUNT(*) FROM news_articles WHERE has_embeddings = true"))
        embedded_count = embedded_articles.scalar()
        
        # Count recent articles (today)
        recent_articles = await session.execute(text("SELECT COUNT(*) FROM news_articles WHERE DATE(created_at) = CURRENT_DATE"))
        recent_count = recent_articles.scalar()
        
        # Get sample of recent articles
        sample_query = text("""
            SELECT title, economic_categories, relevance_score, has_embeddings, vector_db_document_id
            FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        sample_result = await session.execute(sample_query)
        sample_articles = sample_result.fetchall()
    
    print(f"📊 PostgreSQL Storage:")
    print(f"  Total articles: {total_count}")
    print(f"  Articles with embeddings: {embedded_count}")
    print(f"  Articles ingested today: {recent_count}")
    
    # Check ChromaDB embeddings
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Check news collection
        news_collection = chroma_client.get_collection("news_articles")
        news_count = news_collection.count()
        
        # Check economic indicators collection  
        indicators_collection = chroma_client.get_collection("economic_indicators")
        indicators_count = indicators_collection.count()
        
        print(f"\n🔗 ChromaDB Embeddings:")
        print(f"  News articles in ChromaDB: {news_count}")
        print(f"  Economic indicators in ChromaDB: {indicators_count}")
        
        # Test semantic search
        search_results = news_collection.query(
            query_texts=["Federal Reserve interest rates"],
            n_results=3
        )
        
        print(f"\n🔍 Sample Semantic Search (Federal Reserve):")
        for i, (doc_id, distance) in enumerate(zip(search_results['ids'][0], search_results['distances'][0])):
            metadata = search_results['metadatas'][0][i]
            title = metadata.get('title', 'No title')[:60] + "..."
            print(f"  {i+1}. {title} (similarity: {1-distance:.3f})")
            
    except Exception as e:
        print(f"❌ ChromaDB verification failed: {e}")
    
    print(f"\n📰 Sample Recent Articles:")
    for article in sample_articles:
        title = article[0][:50] + "..." if len(article[0]) > 50 else article[0]
        categories = article[1]
        relevance = float(article[2]) if article[2] else 0
        has_embeddings = article[3]
        doc_id = article[4]
        
        status = "✅ Embedded" if has_embeddings else "⏳ Pending"
        print(f"  • {title}")
        print(f"    Categories: {categories}, Relevance: {relevance:.3f}, {status}")

if __name__ == "__main__":
    asyncio.run(verify_storage_and_embeddings())
