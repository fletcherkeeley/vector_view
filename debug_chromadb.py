#!/usr/bin/env python3

import chromadb
import asyncio
from agents.news_sentiment.news_sentiment_data_handler import NewsSentimentDataHandler

async def debug_chromadb():
    print("=== ChromaDB Debug ===")
    
    # Check ChromaDB collections
    client = chromadb.PersistentClient(path='./chroma_db')
    collections = client.list_collections()
    print(f"Collections: {[c.name for c in collections]}")
    
    for collection in collections:
        if 'news' in collection.name.lower():
            count = collection.count()
            print(f"\n{collection.name}: {count} documents")
            
            # Get sample documents
            sample = collection.get(limit=5)
            if sample['documents']:
                print(f"Sample metadata keys: {list(sample['metadatas'][0].keys()) if sample['metadatas'] else 'No metadata'}")
                
                # Test semantic search directly
                print("\n--- Direct ChromaDB Query ---")
                results = collection.query(
                    query_texts=["market volatility economic outlook financial news"],
                    n_results=20
                )
                
                if results and results.get('documents'):
                    print(f"Direct query returned {len(results['documents'][0])} results")
                    for i, doc in enumerate(results['documents'][0][:5]):
                        distance = results['distances'][0][i] if results.get('distances') else 0.0
                        relevance = 1.0 - distance
                        metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                        print(f"  {i+1}. Relevance: {relevance:.3f}, Title: {metadata.get('title', 'No title')[:60]}...")
    
    # Test the data handler
    print("\n=== Testing NewsSentimentDataHandler ===")
    handler = NewsSentimentDataHandler()
    
    articles = await handler.get_news_articles(
        query="market volatility economic outlook financial news",
        max_results=20,
        min_relevance=0.0
    )
    
    print(f"Data handler returned {len(articles)} articles")
    for i, article in enumerate(articles[:5]):
        print(f"  {i+1}. Relevance: {article['relevance_score']:.3f}, Title: {article['title'][:60]}...")

if __name__ == "__main__":
    asyncio.run(debug_chromadb())
