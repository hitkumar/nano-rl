import uvloop
from fastapi import Request

from nano_rl.inference.config import InferenceConfig
from nano_rl.utils.pydantic_config import parse_argv
from vllm.entrypoints.cli.serve import (
    run_api_server_worker_proc as _original_run_api_server_worker_proc,
    run_multi_api_server,
)
from vllm.entrypoints.openai.api_server import engine_client, router, run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _custom_run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Re-import this module in child processes to register custom routes."""
    import nano_rl.inference.server  # noqa: F401

    _original_run_api_server_worker_proc(listen_address, sock, args, client_config, **uvicorn_kwargs)


# Monkey-patch to ensure custom routes work in multi-API-server mode
import vllm.entrypoints.cli.serve

vllm.entrypoints.cli.serve.run_api_server_worker_proc = _custom_run_api_server_worker_proc


@router.post("/update_weights")
async def update_weights(request: Request):
    data = await request.json()
    weight_dir = data.get("weight_dir")
    if not weight_dir:
        return {"status": "error", "message": "weight_dir is missing in the request"}

    await engine_client(request).collective_rpc("update_weights", args=(weight_dir,))
    return {"status": "ok", "weights_dir": weight_dir}


@router.post("/reload_weights")
async def reload_weights(request: Request):
    await engine_client(request).collective_rpc("reload_weights")
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


def main():
    config = parse_argv(InferenceConfig)
    parser = FlexibleArgumentParser(description="nan_rl inference server")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=config.to_vllm_args())
    validate_parsed_serve_args(args)

    # set worker extension
    args.worker_extension_cls = "nano_rl.inference.worker.WeightUpdateWorker"

    if args.api_server_count > 1:
        run_multi_api_server(args)
    else:
        uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
