import pathlib
from typing import Generic, TypeVar

T = TypeVar("T")


class Processor(Generic[T]):
    def process(self, segment_file: pathlib.Path | str) -> T:
        raise NotImplemented
