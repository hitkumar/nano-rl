""" Transport config """

from typing import Literal, TypeAlias

from pydantic import BaseModel


class FileSystemTransportConfig(BaseModel):
    type: Literal["filesystem"] = "filesystem"


TransportConfigType: TypeAlias = FileSystemTransportConfig
