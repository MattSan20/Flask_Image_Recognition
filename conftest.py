"""Shared pytest setup for the tests"""

import pytest
from app import app  # This imports the Flask app for testing

@pytest.fixture
def client():
    """Give each test a Client"""
    with app.test_client() as test_client:
        yield test_client
