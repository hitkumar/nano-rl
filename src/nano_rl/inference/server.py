import os
from argparse import Namespace

import uvloop
from fastapi import APIRouter, Request
from nano_rl.inference.config import InferenceConfig
from nano_rl.utils.pydantic_config import parse_argv
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Create our own router for custom endpoints (vLLM 0.16 no longer exports a shared router)
router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/update_weights")
async def update_weights(request: Request):
    data = await request.json()
    weight_dir = data.get("weight_dir")
    if not weight_dir:
        return {"status": "error", "message": "weight_dir is missing in the request"}

    await engine_client(request).collective_rpc(
        "update_weights_from_path", args=(weight_dir,)
    )
    await engine_client(request).reset_prefix_cache()
    return {"status": "ok", "weights_dir": weight_dir}


@router.post("/reload_weights")
async def reload_weights(request: Request):
    await engine_client(request).collective_rpc("reload_weights")
    await engine_client(request).reset_prefix_cache()
    return {"status": "ok"}


@router.post("/init_broadcaster")
async def init_broadcaster(request: Request):
    data = await request.json()
    host = data.get("host")
    port = data.get("port")
    server_rank = data.get("server_rank")
    num_inference_server = data.get("num_inference_server")
    timeout = data.get("timeout")

    if None in (host, port, server_rank, num_inference_server, timeout):
        return {
            "status": "error",
            "message": "Missing required fields: host, port, server_rank, num_inference_server, timeout",
        }

    await engine_client(request).collective_rpc(
        "init_broadcaster",
        args=(host, port, server_rank, num_inference_server, timeout),
    )
    return {"status": "ok"}


# --- Monkey-patch to inject custom router into vLLM 0.16's app lifecycle ---

import vllm.entrypoints.openai.api_server
from vllm.entrypoints.openai.api_server import build_app as _original_build_app


def custom_build_app(args: Namespace, supported_tasks: tuple):
    """Wrap build_app to include our custom router."""
    app = _original_build_app(args, supported_tasks)
    app.include_router(router)
    return app


vllm.entrypoints.openai.api_server.build_app = custom_build_app


def main():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    config = parse_argv(InferenceConfig)
    parser = FlexibleArgumentParser(description="nano_rl inference server")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=config.to_vllm_args())
    validate_parsed_serve_args(args)

    # set worker extension
    args.worker_extension_cls = "nano_rl.inference.worker.WeightUpdateWorker"

    from vllm.entrypoints.openai.api_server import run_server

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
