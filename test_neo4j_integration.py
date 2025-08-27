#!/usr/bin/env python3
"""
Test script for Neo4j GraphService integration
Verifies connection, basic operations, and Vector View compatibility
"""

import sys
import logging
from datetime import datetime
from database.graph_service import GraphService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    """Test basic Neo4j connection and operations"""
    
    print("🔍 Testing Neo4j GraphService Integration")
    print("=" * 50)
    
    # Initialize GraphService
    graph_service = GraphService()
    
    try:
        # Test connection
        print("1. Testing connection...")
        if graph_service.connect():
            print("✅ Successfully connected to Neo4j")
        else:
            print("❌ Failed to connect to Neo4j")
            return False
        
        # Test database info
        print("\n2. Getting database information...")
        db_info = graph_service.get_database_info()
        print(f"   Nodes: {db_info.get('node_count', 'Unknown')}")
        print(f"   Relationships: {db_info.get('relationship_count', 'Unknown')}")
        print(f"   Labels: {db_info.get('labels', [])}")
        print(f"   Relationship Types: {db_info.get('relationship_types', [])}")
        
        # Test node creation
        print("\n3. Testing node creation...")
        
        # Create an Economic Indicator node
        econ_node = graph_service.create_node(
            label="EconomicIndicator",
            properties={
                "symbol": "GDP",
                "name": "Gross Domestic Product",
                "frequency": "quarterly",
                "source": "FRED",
                "test_run": True
            },
            unique_key="symbol"
        )
        print(f"✅ Created Economic Indicator node: {econ_node.get('symbol')}")
        
        # Create a Market Asset node
        market_node = graph_service.create_node(
            label="MarketAsset",
            properties={
                "symbol": "SPY",
                "name": "SPDR S&P 500 ETF",
                "asset_type": "etf",
                "exchange": "NYSE",
                "test_run": True
            },
            unique_key="symbol"
        )
        print(f"✅ Created Market Asset node: {market_node.get('symbol')}")
        
        # Test relationship creation
        print("\n4. Testing relationship creation...")
        relationship = graph_service.create_relationship(
            from_node_label="EconomicIndicator",
            from_node_key="symbol",
            from_node_value="GDP",
            to_node_label="MarketAsset",
            to_node_key="symbol",
            to_node_value="SPY",
            relationship_type="INFLUENCES",
            properties={
                "correlation": 0.75,
                "confidence": 0.85,
                "analysis_date": datetime.utcnow().isoformat(),
                "test_run": True
            }
        )
        print("✅ Created INFLUENCES relationship between GDP and SPY")
        
        # Test node queries
        print("\n5. Testing node queries...")
        econ_nodes = graph_service.find_nodes("EconomicIndicator", {"test_run": True})
        market_nodes = graph_service.find_nodes("MarketAsset", {"test_run": True})
        print(f"✅ Found {len(econ_nodes)} Economic Indicator nodes")
        print(f"✅ Found {len(market_nodes)} Market Asset nodes")
        
        # Test relationship queries
        print("\n6. Testing relationship queries...")
        relationships = graph_service.find_relationships(
            relationship_type="INFLUENCES",
            from_label="EconomicIndicator",
            to_label="MarketAsset"
        )
        print(f"✅ Found {len(relationships)} INFLUENCES relationships")
        
        # Test custom query
        print("\n7. Testing custom Cypher query...")
        custom_results = graph_service.execute_query("""
            MATCH (e:EconomicIndicator {test_run: true})-[r:INFLUENCES]->(m:MarketAsset {test_run: true})
            RETURN e.symbol as economic_indicator, 
                   m.symbol as market_asset, 
                   r.correlation as correlation,
                   r.confidence as confidence
        """)
        
        for result in custom_results:
            print(f"   {result['economic_indicator']} → {result['market_asset']} "
                  f"(correlation: {result['correlation']}, confidence: {result['confidence']})")
        
        # Cleanup test data
        print("\n8. Cleaning up test data...")
        graph_service.delete_node("EconomicIndicator", "symbol", "GDP")
        graph_service.delete_node("MarketAsset", "symbol", "SPY")
        print("✅ Test data cleaned up")
        
        # Final database info
        print("\n9. Final database state...")
        final_info = graph_service.get_database_info()
        print(f"   Nodes: {final_info.get('node_count', 'Unknown')}")
        print(f"   Relationships: {final_info.get('relationship_count', 'Unknown')}")
        
        print("\n🎉 All Neo4j tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        graph_service.disconnect()
        print("\n🔌 Disconnected from Neo4j")

def test_vector_view_integration():
    """Test Vector View specific graph patterns"""
    
    print("\n🔗 Testing Vector View Integration Patterns")
    print("=" * 50)
    
    graph_service = GraphService()
    
    try:
        if not graph_service.connect():
            print("❌ Failed to connect for integration test")
            return False
        
        # Create sample Vector View entities
        print("1. Creating Vector View entities...")
        
        # Economic indicators from FRED
        fred_indicators = [
            {"symbol": "UNRATE", "name": "Unemployment Rate", "category": "employment"},
            {"symbol": "CPIAUCSL", "name": "Consumer Price Index", "category": "inflation"},
            {"symbol": "FEDFUNDS", "name": "Federal Funds Rate", "category": "monetary_policy"}
        ]
        
        for indicator in fred_indicators:
            graph_service.create_node(
                label="FREDIndicator",
                properties={**indicator, "source": "FRED", "test_integration": True},
                unique_key="symbol"
            )
        
        # Market assets from Yahoo Finance
        market_assets = [
            {"symbol": "^GSPC", "name": "S&P 500", "type": "index"},
            {"symbol": "^VIX", "name": "VIX Volatility Index", "type": "volatility"},
            {"symbol": "GLD", "name": "Gold ETF", "type": "commodity_etf"}
        ]
        
        for asset in market_assets:
            graph_service.create_node(
                label="MarketAsset",
                properties={**asset, "source": "Yahoo", "test_integration": True},
                unique_key="symbol"
            )
        
        # News sentiment nodes
        news_articles = [
            {"article_id": "test_001", "title": "Fed Raises Interest Rates", "sentiment": 0.2, "category": "monetary_policy"},
            {"article_id": "test_002", "title": "Unemployment Drops to New Low", "sentiment": 0.8, "category": "employment"}
        ]
        
        for article in news_articles:
            graph_service.create_node(
                label="NewsArticle",
                properties={**article, "source": "NewsAPI", "test_integration": True},
                unique_key="article_id"
            )
        
        print("✅ Created Vector View test entities")
        
        # Create cross-domain relationships
        print("\n2. Creating cross-domain relationships...")
        
        # Fed Funds Rate influences S&P 500
        graph_service.create_relationship(
            "FREDIndicator", "symbol", "FEDFUNDS",
            "MarketAsset", "symbol", "^GSPC",
            "INFLUENCES",
            {"strength": -0.6, "lag_days": 1, "test_integration": True}
        )
        
        # Unemployment affects VIX
        graph_service.create_relationship(
            "FREDIndicator", "symbol", "UNRATE",
            "MarketAsset", "symbol", "^VIX",
            "CORRELATES_WITH",
            {"correlation": 0.4, "significance": 0.05, "test_integration": True}
        )
        
        # News sentiment affects market
        graph_service.create_relationship(
            "NewsArticle", "article_id", "test_001",
            "MarketAsset", "symbol", "^GSPC",
            "SENTIMENT_IMPACT",
            {"impact_score": -0.3, "time_decay": 24, "test_integration": True}
        )
        
        print("✅ Created cross-domain relationships")
        
        # Test Vector View query patterns
        print("\n3. Testing Vector View query patterns...")
        
        # Find all economic factors affecting S&P 500
        sp500_influences = graph_service.execute_query("""
            MATCH (factor)-[r]->(market:MarketAsset {symbol: '^GSPC', test_integration: true})
            RETURN factor, type(r) as relationship_type, r, market
        """)
        
        print(f"   Found {len(sp500_influences)} factors affecting S&P 500:")
        for influence in sp500_influences:
            # Get the factor node data
            factor_data = influence['factor']
            factor_name = factor_data.get('name', factor_data.get('symbol', 'Unknown'))
            rel_type = influence['relationship_type']
            
            # Determine factor type from the query context
            if 'FRED' in factor_data.get('source', ''):
                factor_type = 'FRED Indicator'
            elif 'News' in factor_data.get('source', ''):
                factor_type = 'News Article'
            else:
                factor_type = 'Market Factor'
                
            print(f"     {factor_type}: {factor_name} --{rel_type}--> S&P 500")
        
        # Find sentiment-market connections
        sentiment_connections = graph_service.execute_query("""
            MATCH (news:NewsArticle {test_integration: true})-[r:SENTIMENT_IMPACT]->(asset:MarketAsset)
            RETURN news.title as headline, 
                   news.sentiment as sentiment,
                   asset.name as asset_name,
                   r.impact_score as impact
        """)
        
        print(f"\n   Found {len(sentiment_connections)} sentiment-market connections:")
        for conn in sentiment_connections:
            print(f"     '{conn['headline']}' (sentiment: {conn['sentiment']}) → "
                  f"{conn['asset_name']} (impact: {conn['impact']})")
        
        # Cleanup integration test data
        print("\n4. Cleaning up integration test data...")
        cleanup_query = """
            MATCH (n {test_integration: true})
            DETACH DELETE n
            RETURN count(n) as deleted_count
        """
        cleanup_result = graph_service.execute_query(cleanup_query)
        deleted_count = cleanup_result[0]['deleted_count'] if cleanup_result else 0
        print(f"✅ Cleaned up {deleted_count} test nodes and relationships")
        
        print("\n🎉 Vector View integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        print(f"❌ Integration test failed: {e}")
        return False
        
    finally:
        graph_service.disconnect()

if __name__ == "__main__":
    print("🚀 Starting Neo4j Integration Tests for Vector View")
    print("Make sure Neo4j is running: docker-compose up neo4j")
    print()
    
    # Run basic tests
    basic_success = test_neo4j_connection()
    
    if basic_success:
        # Run integration tests
        integration_success = test_vector_view_integration()
        
        if integration_success:
            print("\n✅ All tests completed successfully!")
            print("Neo4j is ready for Vector View integration.")
            sys.exit(0)
        else:
            print("\n❌ Integration tests failed")
            sys.exit(1)
    else:
        print("\n❌ Basic connection tests failed")
        print("Please check that Neo4j is running and accessible")
        sys.exit(1)
