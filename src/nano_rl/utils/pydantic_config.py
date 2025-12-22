import sys
from pathlib import Path
from typing import ClassVar, Type, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_settings import (
    BaseSettings as PydanticBaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        # Allows setting None as a string "None" in toml files.
        if v == "None":
            return None
        return v


class BaseSettings(PydanticBaseSettings):
    _TOML_FILES: ClassVar[list[str]] = []

    @classmethod
    def set_toml_files(cls, toml_files: list[str]):
        cls._TOML_FILES = toml_files

    @classmethod
    def clear_toml_files(cls) -> None:
        cls._TOML_FILES = []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            TomlConfigSettingsSource(settings_cls, toml_file=cls._TOML_FILES),
            init_settings,
            env_settings,
        )

    model_config = SettingsConfigDict(
        cli_parse_args=False,
        cli_kebab_case=True,
        nested_model_default_partial_update=True,
    )


def extract_toml_paths(args: list[str]) -> tuple[list[str], list[str]]:
    toml_paths = []
    remaining_args = args.copy()
    while "@" in remaining_args:
        idx = remaining_args.index("@")
        toml_path = remaining_args[idx + 1]
        if not Path(toml_path).exists():
            raise FileNotFoundError(f"Toml file {toml_path} does not exist")

        toml_paths.append(toml_path)
        # remove @ and the toml paths
        remaining_args.pop(idx)
        remaining_args.pop(idx)  # after line 66, toml file is at index 66

    return toml_paths, remaining_args


T = TypeVar("T", bound=BaseSettings)


def parse_argv(config_cls: Type[T]) -> T:
    toml_paths, cli_paths = extract_toml_paths(sys.argv[1:])
    config_cls.set_toml_files(toml_paths)

    try:
        config = config_cls(_cli_parse_args=cli_paths)
        return config
    finally:
        config_cls.clear_toml_files()


# scratch pad
# class ModelConfig(BaseModel):
#     name: str
#     learning_rate: float = 0.001
#     batch_size: int = 32


# config = ModelConfig(name="nano-rl", learning_rate=0.01, batch_size=64)
# print(config)
