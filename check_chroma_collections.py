#!/usr/bin/env python3
"""
Check ChromaDB collections and their contents
"""
import chromadb
from chromadb.config import Settings
import os

def check_collections():
    """Check what collections exist in ChromaDB"""
    
    chroma_persist_dir = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # List all collections
        collections = client.list_collections()
        
        print(f"🔍 ChromaDB Collections in {chroma_persist_dir}:")
        print(f"Found {len(collections)} collections:")
        
        for collection in collections:
            print(f"\n📁 Collection: {collection.name}")
            print(f"   ID: {collection.id}")
            print(f"   Metadata: {collection.metadata}")
            
            # Get count
            try:
                count = collection.count()
                print(f"   Documents: {count:,}")
                
                # Get a few sample documents if any exist
                if count > 0:
                    results = collection.peek(limit=3)
                    print(f"   Sample IDs: {results['ids'][:3] if results['ids'] else 'None'}")
                    
            except Exception as e:
                print(f"   Error getting count: {e}")
        
        return collections
        
    except Exception as e:
        print(f"❌ Error checking ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    check_collections()
