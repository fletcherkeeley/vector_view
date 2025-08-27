#!/usr/bin/env python3
"""
ChromaDB Embedding Quality Validation

Analyzes similarity score distribution across financial keywords to validate
embedding quality and determine appropriate thresholds.
"""

import sys
sys.path.append('/home/lab/projects/vector-view')

import chromadb
from chromadb.config import Settings
import numpy as np
from collections import defaultdict

def analyze_chromadb_embeddings():
    """Analyze ChromaDB embedding quality and similarity score distribution."""
    
    print("🔍 CHROMADB EMBEDDING QUALITY ANALYSIS")
    print("=" * 50)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path='/home/lab/projects/vector-view/chroma_db',
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_collection('financial_news')
    print(f"Total articles in ChromaDB: {collection.count()}")
    
    # Test queries with different financial keywords
    test_queries = [
        "Federal Reserve",
        "interest rates", 
        "inflation",
        "employment",
        "GDP growth",
        "stock market",
        "earnings report",
        "monetary policy",
        "economic indicators",
        "recession"
    ]
    
    results_summary = {}
    
    for query in test_queries:
        print(f"\n📊 Testing query: '{query}'")
        
        # Get top 10 results for this query
        results = collection.query(
            query_texts=[query],
            n_results=10,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            print(f"   ❌ No results found")
            continue
            
        similarities = [1.0 - dist for dist in results['distances'][0]]
        
        print(f"   📈 Similarity scores: {[f'{s:.3f}' for s in similarities[:5]]}")
        print(f"   📊 Range: {min(similarities):.3f} - {max(similarities):.3f}")
        print(f"   📊 Average: {np.mean(similarities):.3f}")
        
        # Show top 3 articles for manual validation
        print(f"   🔍 Top 3 matches:")
        for i in range(min(3, len(results['documents'][0]))):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            similarity = similarities[i]
            
            # Extract title from document content
            title = "No title"
            if doc.startswith("Title: "):
                title_end = doc.find(" Content: ")
                if title_end > 0:
                    title = doc[7:title_end]
            
            categories = metadata.get('economic_categories', 'N/A')
            source = metadata.get('source_name', 'N/A')
            
            print(f"     {i+1}. Similarity: {similarity:.3f}")
            print(f"        Title: {title[:60]}...")
            print(f"        Categories: {categories}")
            print(f"        Source: {source}")
            
        results_summary[query] = {
            'count': len(similarities),
            'max_similarity': max(similarities),
            'avg_similarity': np.mean(similarities),
            'min_similarity': min(similarities),
            'top_3_similarities': similarities[:3]
        }
    
    # Overall analysis
    print(f"\n📋 OVERALL SIMILARITY ANALYSIS")
    print("=" * 50)
    
    all_max_scores = [r['max_similarity'] for r in results_summary.values()]
    all_avg_scores = [r['avg_similarity'] for r in results_summary.values()]
    
    print(f"Best similarity scores across queries:")
    print(f"  Highest: {max(all_max_scores):.3f}")
    print(f"  Average of best: {np.mean(all_max_scores):.3f}")
    print(f"  Lowest best: {min(all_max_scores):.3f}")
    
    print(f"\nAverage similarity scores across queries:")
    print(f"  Highest avg: {max(all_avg_scores):.3f}")
    print(f"  Overall avg: {np.mean(all_avg_scores):.3f}")
    print(f"  Lowest avg: {min(all_avg_scores):.3f}")
    
    # Threshold recommendations
    print(f"\n🎯 THRESHOLD RECOMMENDATIONS")
    print("=" * 50)
    
    conservative_threshold = np.percentile(all_max_scores, 25)  # 25th percentile of best scores
    moderate_threshold = np.percentile(all_max_scores, 10)     # 10th percentile of best scores
    liberal_threshold = np.percentile(all_avg_scores, 50)     # Median of average scores
    
    print(f"Conservative (high precision): {conservative_threshold:.3f}")
    print(f"Moderate (balanced): {moderate_threshold:.3f}")
    print(f"Liberal (high recall): {liberal_threshold:.3f}")
    
    # Check for potential embedding issues
    print(f"\n🔧 EMBEDDING QUALITY ASSESSMENT")
    print("=" * 50)
    
    if max(all_max_scores) < 0.3:
        print("⚠️  WARNING: Very low similarity scores detected")
        print("   This may indicate embedding quality issues or mismatched models")
    elif max(all_max_scores) < 0.5:
        print("⚠️  CAUTION: Lower than expected similarity scores")
        print("   Consider investigating embedding model or data preprocessing")
    else:
        print("✅ Similarity scores appear reasonable")
    
    # Specific Fed analysis
    if 'Federal Reserve' in results_summary:
        fed_results = results_summary['Federal Reserve']
        print(f"\n🏛️  FEDERAL RESERVE SPECIFIC ANALYSIS")
        print(f"   Best Fed similarity: {fed_results['max_similarity']:.3f}")
        print(f"   Avg Fed similarity: {fed_results['avg_similarity']:.3f}")
        
        if fed_results['max_similarity'] > 0.2:
            print("   ✅ Fed content appears well-embedded")
        else:
            print("   ⚠️  Fed content may have embedding issues")
    
    return results_summary, {
        'conservative': conservative_threshold,
        'moderate': moderate_threshold, 
        'liberal': liberal_threshold
    }

if __name__ == "__main__":
    results, thresholds = analyze_chromadb_embeddings()
    
    print(f"\n🎉 ANALYSIS COMPLETE")
    print(f"Recommended verification threshold: {thresholds['moderate']:.3f}")
