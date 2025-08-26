#!/usr/bin/env python3

import chromadb
import asyncio

async def debug_semantic_search():
    print("=== Semantic Search Debug ===")
    
    client = chromadb.PersistentClient(path='../../chroma_db')
    collection = client.get_collection("financial_news")
    
    print(f"Total documents: {collection.count()}")
    
    # Test different queries and thresholds
    queries = [
        "market volatility economic outlook financial news",
        "market",
        "financial",
        "economy",
        "stock market",
        "economic indicators"
    ]
    
    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        results = collection.query(
            query_texts=[query],
            n_results=50
        )
        
        if results and results.get('documents'):
            total_results = len(results['documents'][0])
            positive_relevance = 0
            
            for i, distance in enumerate(results['distances'][0]):
                relevance = 1.0 - distance
                if relevance > 0:
                    positive_relevance += 1
                if i < 5:  # Show first 5
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    print(f"  {i+1}. Relevance: {relevance:.3f}, Source: {metadata.get('source_name', 'unknown')}")
            
            print(f"  Total results: {total_results}, Positive relevance: {positive_relevance}")
    
    # Check metadata distribution
    print(f"\n--- Sample Metadata ---")
    sample = collection.get(limit=10)
    if sample['metadatas']:
        for i, meta in enumerate(sample['metadatas'][:3]):
            print(f"  Article {i+1}: {dict(meta)}")

if __name__ == "__main__":
    asyncio.run(debug_semantic_search())
