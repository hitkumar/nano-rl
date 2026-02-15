"""
vLLM monkey-patches for nano_rl compatibility.
"""

from argparse import Namespace


def monkey_patch_build_app():
    """Patch build_app to include nano_rl's custom router (weight update endpoints)."""
    import vllm.entrypoints.openai.api_server
    from vllm.entrypoints.openai.api_server import build_app as _original_build_app

    from nano_rl.inference.server import router

    def custom_build_app(args: Namespace, supported_tasks: tuple):
        app = _original_build_app(args, supported_tasks)
        app.include_router(router)
        return app

    vllm.entrypoints.openai.api_server.build_app = custom_build_app


def monkey_patch_tokenize_params_validation():
    """Patch TokenizeParams to only reject prompts exceeding max_model_len,
    not prompts where prompt_len + max_tokens > max_model_len.

    Without this, multi-turn prompts get rejected because vLLM reserves
    max_tokens of headroom even though the model may generate far fewer tokens.
    """
    from vllm.exceptions import VLLMValidationError
    from vllm.renderers.params import TokenizeParams

    def _patched_token_len_check(self, tokenizer, tokens):
        if self.max_total_tokens is not None and len(tokens) > self.max_total_tokens:
            raise VLLMValidationError(
                f"The prompt is {len(tokens)} tokens, which exceeds the "
                f"model's maximum context length of {self.max_total_tokens} tokens. "
                f"Please reduce the length of the input prompt.",
                parameter="input_tokens",
                value=len(tokens),
            )
        return tokens

    def _patched_text_len_check(self, tokenizer, text):
        if self.max_total_tokens is None or tokenizer is None:
            return text
        if self.truncate_prompt_tokens is None:
            max_chars = self.max_total_tokens * tokenizer.max_chars_per_token
            if len(text) > max_chars:
                raise VLLMValidationError(
                    f"You passed {len(text)} input characters. "
                    f"However, the model's context length is only "
                    f"{self.max_total_tokens} tokens "
                    f"(at most {max_chars} characters). "
                    f"Please reduce the length of the input prompt.",
                    parameter="input_text",
                    value=len(text),
                )
        return text

    TokenizeParams._token_len_check = _patched_token_len_check
    TokenizeParams._text_len_check = _patched_text_len_check
