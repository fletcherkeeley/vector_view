"""
Neo4j Graph Database Service for Vector View Financial Intelligence Platform

Provides graph database operations for relationship modeling between:
- Economic indicators and market movements
- News sentiment and asset correlations
- Cross-domain financial intelligence connections

Features:
- Connection management with retry logic
- Basic CRUD operations for nodes and relationships
- Query execution with error handling
- Integration with existing Vector View architecture
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime

from neo4j import GraphDatabase, Driver, Session, Result
from neo4j.exceptions import ServiceUnavailable, TransientError, ClientError

# Configure logging
logger = logging.getLogger(__name__)

class GraphService:
    """
    Neo4j Graph Database Service for Vector View platform
    
    Handles connections, transactions, and basic graph operations
    following Vector View's database patterns and error handling.
    """
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = "neo4j"
    ):
        """
        Initialize GraphService with connection parameters
        
        Args:
            uri: Neo4j connection URI (default: bolt://localhost:7687)
            username: Neo4j username (default: neo4j)
            password: Neo4j password (default: vector_view_password)
            database: Neo4j database name (default: neo4j)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "vector_view_password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        self._driver: Optional[Driver] = None
        self._connected = False
        
        logger.info(f"GraphService initialized for {self.uri}")
    
    def connect(self) -> bool:
        """
        Establish connection to Neo4j database
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Verify connectivity
            self._driver.verify_connectivity()
            self._connected = True
            
            logger.info("Successfully connected to Neo4j database")
            return True
            
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close Neo4j driver connection"""
        if self._driver:
            self._driver.close()
            self._connected = False
            logger.info("Disconnected from Neo4j database")
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j"""
        return self._connected and self._driver is not None
    
    @asynccontextmanager
    async def get_session(self):
        """
        Async context manager for Neo4j sessions
        
        Usage:
            async with graph_service.get_session() as session:
                result = session.run("MATCH (n) RETURN count(n)")
        """
        if not self.is_connected():
            if not self.connect():
                raise ConnectionError("Failed to connect to Neo4j database")
        
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            timeout: Query timeout in seconds
            
        Returns:
            List of result records as dictionaries
        """
        if not self.is_connected():
            if not self.connect():
                raise ConnectionError("Failed to connect to Neo4j database")
        
        parameters = parameters or {}
        
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, parameters, timeout=timeout)
                records = [record.data() for record in result]
                
                logger.debug(f"Query executed successfully: {len(records)} records returned")
                return records
                
        except TransientError as e:
            logger.warning(f"Transient error in query execution, retrying: {e}")
            # Retry once for transient errors
            try:
                with self._driver.session(database=self.database) as session:
                    result = session.run(query, parameters, timeout=timeout)
                    records = [record.data() for record in result]
                    return records
            except Exception as retry_error:
                logger.error(f"Query retry failed: {retry_error}")
                raise
                
        except ClientError as e:
            logger.error(f"Client error in query execution: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in query execution: {e}")
            raise
    
    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        unique_key: str = None
    ) -> Dict[str, Any]:
        """
        Create a node with specified label and properties
        
        Args:
            label: Node label (e.g., 'EconomicIndicator', 'NewsArticle')
            properties: Node properties dictionary
            unique_key: Property name for uniqueness constraint
            
        Returns:
            Created node data
        """
        # Add timestamp if not provided
        if 'created_at' not in properties:
            properties['created_at'] = datetime.utcnow().isoformat()
        
        if unique_key and unique_key in properties:
            # Use MERGE for unique nodes
            query = f"""
            MERGE (n:{label} {{{unique_key}: ${unique_key}}})
            SET n += $properties
            RETURN n
            """
            parameters = {
                unique_key: properties[unique_key],
                'properties': properties
            }
        else:
            # Use CREATE for non-unique nodes
            query = f"CREATE (n:{label} $properties) RETURN n"
            parameters = {'properties': properties}
        
        results = self.execute_query(query, parameters)
        
        if results:
            logger.info(f"Created {label} node with properties: {list(properties.keys())}")
            return results[0]['n']
        else:
            raise RuntimeError(f"Failed to create {label} node")
    
    def create_relationship(
        self,
        from_node_label: str,
        from_node_key: str,
        from_node_value: Any,
        to_node_label: str,
        to_node_key: str,
        to_node_value: Any,
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes
        
        Args:
            from_node_label: Source node label
            from_node_key: Source node property key for matching
            from_node_value: Source node property value for matching
            to_node_label: Target node label
            to_node_key: Target node property key for matching
            to_node_value: Target node property value for matching
            relationship_type: Relationship type (e.g., 'CORRELATES_WITH', 'INFLUENCES')
            properties: Relationship properties dictionary
            
        Returns:
            Created relationship data
        """
        properties = properties or {}
        
        # Add timestamp if not provided
        if 'created_at' not in properties:
            properties['created_at'] = datetime.utcnow().isoformat()
        
        query = f"""
        MATCH (from:{from_node_label} {{{from_node_key}: $from_value}})
        MATCH (to:{to_node_label} {{{to_node_key}: $to_value}})
        MERGE (from)-[r:{relationship_type}]->(to)
        SET r += $properties
        RETURN r, from, to
        """
        
        parameters = {
            'from_value': from_node_value,
            'to_value': to_node_value,
            'properties': properties
        }
        
        results = self.execute_query(query, parameters)
        
        if results:
            logger.info(f"Created {relationship_type} relationship")
            return results[0]
        else:
            raise RuntimeError(f"Failed to create {relationship_type} relationship")
    
    def find_nodes(
        self,
        label: str,
        properties: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by label and optional properties
        
        Args:
            label: Node label to search for
            properties: Optional properties to match
            limit: Maximum number of results
            
        Returns:
            List of matching nodes
        """
        if properties:
            where_clauses = [f"n.{key} = ${key}" for key in properties.keys()]
            where_clause = "WHERE " + " AND ".join(where_clauses)
            parameters = properties
        else:
            where_clause = ""
            parameters = {}
        
        query = f"""
        MATCH (n:{label})
        {where_clause}
        RETURN n
        LIMIT {limit}
        """
        
        results = self.execute_query(query, parameters)
        return [result['n'] for result in results]
    
    def find_relationships(
        self,
        relationship_type: str = None,
        from_label: str = None,
        to_label: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find relationships with optional filters
        
        Args:
            relationship_type: Optional relationship type filter
            from_label: Optional source node label filter
            to_label: Optional target node label filter
            limit: Maximum number of results
            
        Returns:
            List of matching relationships with connected nodes
        """
        # Build match pattern
        from_pattern = f"(from:{from_label})" if from_label else "(from)"
        to_pattern = f"(to:{to_label})" if to_label else "(to)"
        rel_pattern = f"[r:{relationship_type}]" if relationship_type else "[r]"
        
        query = f"""
        MATCH {from_pattern}-{rel_pattern}->{to_pattern}
        RETURN r, from, to
        LIMIT {limit}
        """
        
        results = self.execute_query(query)
        return results
    
    def delete_node(self, label: str, key: str, value: Any) -> bool:
        """
        Delete a node and all its relationships
        
        Args:
            label: Node label
            key: Property key for matching
            value: Property value for matching
            
        Returns:
            True if node was deleted, False if not found
        """
        query = f"""
        MATCH (n:{label} {{{key}: $value}})
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        
        results = self.execute_query(query, {'value': value})
        deleted_count = results[0]['deleted_count'] if results else 0
        
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} {label} node(s)")
            return True
        else:
            logger.warning(f"No {label} node found with {key}={value}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get basic database information and statistics
        
        Returns:
            Dictionary with database statistics
        """
        queries = {
            'node_count': "MATCH (n) RETURN count(n) as count",
            'relationship_count': "MATCH ()-[r]->() RETURN count(r) as count",
            'labels': "CALL db.labels() YIELD label RETURN collect(label) as labels",
            'relationship_types': "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        }
        
        info = {}
        for key, query in queries.items():
            try:
                result = self.execute_query(query)
                if key == 'labels':
                    info[key] = result[0]['labels'] if result else []
                elif key == 'relationship_types':
                    info[key] = result[0]['types'] if result else []
                else:
                    info[key] = result[0]['count'] if result else 0
            except Exception as e:
                logger.warning(f"Failed to get {key}: {e}")
                info[key] = None
        
        return info
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Convenience function for creating a GraphService instance
def create_graph_service(
    uri: str = None,
    username: str = None,
    password: str = None,
    database: str = "neo4j"
) -> GraphService:
    """
    Create and return a configured GraphService instance
    
    Args:
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        database: Neo4j database name
        
    Returns:
        Configured GraphService instance
    """
    return GraphService(uri, username, password, database)
