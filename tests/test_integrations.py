"""
Basic tests for poma_integrations package.
"""
import pytest

def test_imports():
    """Test that all components can be imported."""
    from poma_integrations import (
        Doc2PomaLoader,
        PomaSentenceSplitter,
        PomaChunksetSplitter,
        PomaCheatsheetRetriever,
        Doc2PomaReader,
        PomaSentenceNodeParser,
        PomaChunksetNodeParser,
        PomaCheatsheetPostProcessor,
    )
    
    assert Doc2PomaLoader
    assert PomaSentenceSplitter
    assert PomaChunksetSplitter
    assert PomaCheatsheetRetriever
    assert Doc2PomaReader
    assert PomaSentenceNodeParser
    assert PomaChunksetNodeParser
    assert PomaCheatsheetPostProcessor
