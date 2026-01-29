"""
Client util for orchestrtor -> inference communication
"""

import asyncio
from pathlib import Path

import httpx
from httpx import AsyncClient
from loguru import logger
from nano_rl.utils.config import ClientConfig
from openai import AsyncOpenAI


def setup_clients(client_config: ClientConfig) -> list[AsyncOpenAI]:
    """Create OpenAI clients for all inference servers."""
    timeout = httpx.Timeout(client_config.timeout)
    limits = httpx.Limits(max_connections=8192, max_keepalive_connections=8192)

    def _setup_client(base_url: str) -> AsyncOpenAI:
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            max_retries=10,
            http_client=http_client,
        )

    return [_setup_client(url) for url in client_config.base_url]


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create admin clients for all inference servers."""

    def _setup_admin_client(base_url: str) -> AsyncClient:
        base_url = base_url.rstrip("/").removesuffix("/v1")
        return AsyncClient(
            base_url=base_url,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            timeout=httpx.Timeout(client_config.timeout),
        )

    return [_setup_admin_client(url) for url in client_config.base_url]


async def check_health(admin_clients: list[AsyncClient], timeout: int = 1800) -> None:
    """Wait for all inference servers to be ready"""

    async def _check_health(admin_client: AsyncClient) -> None:
        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                response = await admin_client.get("/health")
                if response.status_code == 200:
                    logger.debug(f"Inference server {admin_client.base_url} is ready")
                    return
            except httpx.RequestError:
                pass

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(
                    f"Inference server {admin_client.base_url} not started after {timeout}s"
                )
            await asyncio.sleep(1)

    await asyncio.gather(*[_check_health(client) for client in admin_clients])
    logger.info(f"All {len(admin_clients)} inference server(s) ready")


async def update_weights(
    admin_clients: list[AsyncClient], weight_dir: Path | None
) -> None:
    """Update model weights on all inference servers"""

    async def _update_weights(admin_client: AsyncClient) -> None:
        body = {"weight_dir": str(weight_dir)} if weight_dir else {}
        response = await admin_client.post("/update_weights", json=body)
        response.raise_for_status()

    await asyncio.gather(*[_update_weights(client) for client in admin_clients])
    logger.info(f"Updated weights to {weight_dir}")


async def reload_weights(admin_clients: list[AsyncClient]) -> None:
    """Reloads base model weights on all inference servers"""

    async def _reload_weights(admin_client: AsyncClient) -> None:
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

    await asyncio.gather(*[_reload_weights(client) for client in admin_clients])
    logger.info("Reloaded weights")
