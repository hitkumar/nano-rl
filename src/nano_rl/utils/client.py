"""
Client util for orchestrtor -> inference communication
"""

import asyncio
import json
from pathlib import Path

import httpx
from httpx import AsyncClient
from loguru import logger
from nano_rl.utils.config import ClientConfig
from openai import AsyncOpenAI


def setup_client(client_config: ClientConfig) -> AsyncOpenAI:
    # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
    # This client is used for LLM inference
    timeout = httpx.Timeout(client_config.timeout)
    limits = httpx.Limits(max_connections=8192, max_keepalive_connections=8192)
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    return AsyncOpenAI(
        base_url=client_config.base_url,
        # do not need a real key as we connect with local vllm server.
        api_key="EMPTY",
        max_retries=10,
        http_client=http_client,
    )


def setup_admin_client(client_config: ClientConfig) -> AsyncClient:
    base_url = client_config.base_url.rstrip("/").removesuffix("/v1")
    return AsyncClient(
        base_url=base_url,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        timeout=httpx.Timeout(client_config.timeout),
    )


async def check_health(admin_client: AsyncClient, timeout: int = 1800) -> None:
    """Wait for inference server to be ready"""
    start_time = asyncio.get_event_loop().time()
    while True:
        try:
            response = await admin_client.get("/health")
            if response.status_code == 200:
                logger.info("Inference server is ready")
                return
        except httpx.RequestError:
            pass

        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError(f"Inference server not started after {timeout}s")

        # sleep one sec before attempting to connect again.
        await asyncio.sleep(1)


async def update_weights(admin_client: AsyncClient, weight_dir: Path | None) -> None:
    """Update model weights on inference server"""
    body = {"weight_dir": str(weight_dir)} if weight_dir else {}
    response = await admin_client.post("/update_weights", json=body)
    response.raise_for_status()
    logger.info(f"Updated weights to {weight_dir}")


async def reload_weights(admin_client: AsyncClient) -> None:
    """Reloads base model weights"""
    try:
        response = await admin_client.post("/reload_weights", json={})
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(
                "The route /reload_weights does not exist. Skipping weight update."
            )
            return
        raise
    logger.info("Reloaded weights")
