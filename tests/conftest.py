"""Pytest configuration and fixtures"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Generate a sample RGB image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_batch():
    """Generate a batch of sample images."""
    return np.random.randint(0, 255, (4, 480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def config_dir():
    """Return path to config directory."""
    return Path(__file__).parent.parent / "config"
