#!/usr/bin/env python3
"""
Test CLI for Semantic Search System

Simple command-line interface to test the semantic search capabilities
before integrating with AI agents.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "database"))

from dotenv import load_dotenv
from .embedding_pipeline import create_embedding_pipeline
from .search_interface import AgentSearchInterface, create_agent_search_interface

load_dotenv()


async def test_embedding_pipeline():
    """Test the embedding pipeline"""
    print("🔄 Testing Embedding Pipeline...")
    
    try:
        pipeline = await create_embedding_pipeline()
        stats = await pipeline.get_pipeline_stats()
        
        print(f"📊 Pipeline Stats:")
        print(f"   • Total Articles: {stats['postgresql']['total_articles']}")
        print(f"   • Embedded Articles: {stats['postgresql']['embedded_articles']}")
        print(f"   • Completion: {stats['completion_percentage']:.1f}%")
        print(f"   • Vector Collections: {list(stats['vector_store'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


async def test_search_interface():
    """Test the search interface"""
    print("\n🔍 Testing Search Interface...")
    
    try:
        interface = await create_agent_search_interface()
        
        # Test queries
        test_queries = [
            "Federal Reserve interest rate decisions",
            "unemployment data trends",
            "market volatility news",
            "inflation impact on economy"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            
            result = await interface.search_with_context(
                query=query,
                agent_id="test_cli",
                max_results=5
            )
            
            print(f"   Intent: {result['intent']}")
            print(f"   Total Results: {result['total_results']}")
            
            for collection, items in result['results'].items():
                if items:
                    print(f"   {collection}: {len(items)} results")
                    # Show top result
                    top_result = items[0]
                    score = top_result.get('final_score', 0)
                    print(f"     Top: {score:.3f} - {top_result['document'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


async def run_embedding_pipeline():
    """Run the full embedding pipeline"""
    print("🚀 Running Full Embedding Pipeline...")
    
    try:
        pipeline = await create_embedding_pipeline()
        results = await pipeline.run_full_embedding_pipeline()
        
        print(f"✅ Pipeline Complete!")
        print(f"   • News Articles Processed: {results['news_articles']['processed']}")
        print(f"   • News Articles Failed: {results['news_articles']['failed']}")
        print(f"   • Economic Indicators Processed: {results['economic_indicators']['processed']}")
        print(f"   • Economic Indicators Failed: {results['economic_indicators']['failed']}")
        print(f"   • Duration: {results['duration_seconds']:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return False


async def interactive_search():
    """Interactive search mode"""
    print("\n🤖 Interactive Search Mode (type 'quit' to exit)")
    
    try:
        interface = await create_agent_search_interface()
        
        while True:
            query = input("\n🔍 Enter search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            result = await interface.search_with_context(
                query=query,
                agent_id="interactive_cli",
                max_results=10
            )
            
            print(f"\n📊 Results for: '{query}'")
            print(f"Intent: {result['intent']} | Total: {result['total_results']}")
            
            for collection, items in result['results'].items():
                if items:
                    print(f"\n📁 {collection.upper()} ({len(items)} results):")
                    for i, item in enumerate(items[:3], 1):  # Show top 3
                        score = item.get('final_score', 0)
                        doc_preview = item['document'][:150] + "..." if len(item['document']) > 150 else item['document']
                        print(f"   {i}. [{score:.3f}] {doc_preview}")
        
        print("👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Interactive search failed: {e}")


def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("🧠 VECTOR VIEW SEMANTIC SEARCH SYSTEM")
    print("="*60)
    print("1. Test System Status")
    print("2. Run Embedding Pipeline")
    print("3. Test Search Interface")
    print("4. Interactive Search")
    print("5. Exit")
    print("="*60)


async def main():
    """Main CLI function"""
    print("🚀 Vector View Semantic Search Test CLI")
    
    while True:
        print_menu()
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            await test_embedding_pipeline()
            await test_search_interface()
        
        elif choice == "2":
            await run_embedding_pipeline()
        
        elif choice == "3":
            await test_search_interface()
        
        elif choice == "4":
            await interactive_search()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    asyncio.run(main())
