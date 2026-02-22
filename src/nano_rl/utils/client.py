"""
Client util for orchestrator -> inference communication
"""

import asyncio
from pathlib import Path

import httpx
import verifiers as vf
from httpx import AsyncClient
from loguru import logger
from nano_rl.utils.config import ClientConfig


NCCL_READY_MARKER = "NCCL_READY"


def setup_clients(client_config: ClientConfig) -> list[vf.ClientConfig]:
    """Create verifiers ClientConfig for all inference servers."""

    def _setup_client(client_idx: int, base_url: str) -> vf.ClientConfig:
        return vf.ClientConfig(
            client_idx=client_idx,
            client_type="openai_chat_completions",
            api_base_url=base_url,
            api_key_var="OPENAI_API_KEY",
            timeout=client_config.timeout,
            max_connections=8192,
            max_keepalive_connections=8192,
            max_retries=10,
        )

    return [_setup_client(idx, url) for idx, url in enumerate(client_config.base_url)]


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

    if weight_dir is not None:
        nccl_ready_file = weight_dir / NCCL_READY_MARKER
        nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
        nccl_ready_file.touch()
        logger.debug(f"Creating nccl ready file at {nccl_ready_file}")

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


async def init_nccl_broadcast(
    admin_clients: list[AsyncClient], host: str, port: int, timeout: int
) -> None:
    """Initialize NCCL broadcast receivers on all inference servers."""

    async def _init_nccl_broadcast(admin_client: AsyncClient, server_rank: int) -> None:
        try:
            response = await admin_client.post(
                "/init_broadcaster",
                json={
                    "host": host,
                    "port": port,
                    "server_rank": server_rank,
                    "num_inference_server": len(admin_clients),
                    "timeout": timeout,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(
                    "The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization."
                )
                return
            raise

    await asyncio.gather(
        *[
            _init_nccl_broadcast(admin_client, server_rank)
            for server_rank, admin_client in enumerate(admin_clients)
        ]
    )
    logger.info(
        f"Initialized NCCL broadcast on {len(admin_clients)} inference server(s)"
    )
