"""Opt-in live connectivity tests for SearXNG and Trafilatura extraction."""

import os

import aiohttp
import pytest
from trafilatura import extract, extract_metadata

pytestmark = pytest.mark.live
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")


@pytest.mark.asyncio
async def test_searxng_live_search():
    """Verify the configured SearXNG instance returns JSON search results."""
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            f"{SEARXNG_URL}/search",
            params={"q": "python", "format": "json"},
            headers={"X-Real-IP": "127.0.0.1"},
        ) as response:
            response.raise_for_status()
            payload = await response.json()

    assert payload.get("results"), "SearXNG returned no results."
    assert payload["results"][0].get("url")


@pytest.mark.asyncio
async def test_trafilatura_live_extraction():
    """Verify Trafilatura extracts article text and metadata from a live page."""
    url = "https://www.gov.uk/"
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            html = await response.text()

    content = extract(html, url=url)
    metadata = extract_metadata(html)

    assert content, f"Trafilatura extracted no content from {url}."
    assert metadata and metadata.title, f"Trafilatura extracted no title from {url}."
