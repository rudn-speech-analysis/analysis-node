import requests
import tempfile
import pathlib
import json
import numpy as np
from dataclasses import fields, is_dataclass


# Source - https://stackoverflow.com/a/34073559
# Posted by Ferdinand Beyer, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-08, License - CC BY-SA 4.0
class GeneratorReturnCatcher:
    """
    Used to receive the return value from a generator.
    Use as follows:

    ```
    def foobar():
        yield 1
        yield 2
        return 3

    gen = GeneratorReturnCatcher(foobar())
    for x in gen:
        print('item', x)
    
    print('return', gen.value)

    # prints: item 1, item 2, return 3
    ```
    """
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
        return self.value


def fetch_to_tmp_file(url: str, file_extension="") -> tempfile._TemporaryFileWrapper:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    tmp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False, delete_on_close=False)
    for chunk in response.iter_content(chunk_size=8192):
        tmp_file.write(chunk)
    tmp_file.flush()
    tmp_file.seek(0)

    return tmp_file


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def list_dict_to_dict_list(data: list[dict]) -> dict[str, list]:
    """
    list[dict[str, Any]] -> dict[str, list[Any]]
    list[dataclass[str, num]] -> dict[str, list[num]]
    """

    if not data:
        return dict()
    if is_dataclass(data[0]):
        field_names = [field.name for field in fields(data[0])]
        return {name: [getattr(obj, name) for obj in data] for name in field_names}
    else:
        return {key: [d[key] for d in data] for key in data[0]}
