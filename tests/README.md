# Test Scripts

This directory contains all test scripts for the Vector View platform.

## Directory Structure

### `/agents/`
Agent-specific tests:
- `test_agents.py` - Core agent functionality tests
- `test_ai_agents.py` - AI service integration tests
- `test_all_agents_fixed.py` - Comprehensive agent validation
- `test_agent_dependency_flow.py` - Agent dependency and workflow tests

### `/individual/`
Individual component tests:
- `test_market_intelligence_individual.py` - Market intelligence agent tests
- `test_news_sentiment_individual.py` - News sentiment analysis tests
- `test_single_agent.py` - Single agent isolation tests

### `/integration/`
Integration and system-wide tests:
- `test_ai_service_fix.py` - AI service integration fixes
- `test_network_fixes.py` - Network connectivity tests
- `test_wsj_agents.py` - WSJ-level analysis integration tests

## Running Tests

```bash
# Run all agent tests
python -m pytest tests/agents/

# Run individual component tests
python -m pytest tests/individual/

# Run integration tests
python -m pytest tests/integration/

# Run specific test file
python tests/agents/test_agents.py
```

## Test Categories

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Multi-component interactions
- **Agent Tests**: AI agent behavior and responses
- **System Tests**: End-to-end workflow validation
