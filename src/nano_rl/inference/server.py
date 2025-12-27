import uvloop
from fastapi import Request

from nano_rl.inference.config import InferenceConfig
from nano_rl.utils.pydantic_config import parse_argv
from vllm.entrypoints.openai.api_server import engine_client, router, run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser


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


def main():
    config = parse_argv(InferenceConfig)
    parser = FlexibleArgumentParser(description="nan_rl inference server")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=config.to_vllm_args())
    validate_parsed_serve_args(args)

    # set worker extension
    args.worker_extension_cls = "nano_rl.inference.worker.WeightUpdateWorker"
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
