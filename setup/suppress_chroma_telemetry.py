#!/usr/bin/env python3
"""
Suppress ChromaDB telemetry errors by setting environment variables
"""

import os
import logging

# Suppress ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'True'

# Suppress specific ChromaDB logging
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

print("ChromaDB telemetry suppression configured")
