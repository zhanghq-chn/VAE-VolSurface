import yaml
from typing import Any, Dict, Iterator, Union
from pathlib import Path
import re
from itertools import product
from collections.abc import Iterable


class YamlParser:
    """
    A utility class for parsing and manipulating YAML files.

    Methods:
        __init__(file_path: Union[str, Path]) -> None:
            Initializes the YamlParser with the path to the YAML file.
        load_yaml() -> Dict[str, Any]:
            Loads the YAML file and returns its contents as a dictionary.
        save_yaml(data: Dict[str, Any]) -> None:
            Saves the provided dictionary to the YAML file.
        load_yaml_matrix() -> Iterator[Dict[str, Any]]:
            Loads a matrix from a YAML file and yields dictionaries representing all possible combinations of the matrix values.
        force_iterable(data: dict[Any, Any]) -> dict[Any, Any]:
            Recursively converts all dictionary values to lists if they are not already lists.
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = file_path

    def load_yaml(self) -> Dict[str, Any]:
        with open(self.file_path, "r") as file:
            data: Dict[str, Any] = yaml.safe_load(file)
        return data

    def save_yaml(self, data: Dict[str, Any]) -> None:
        with open(self.file_path, "w") as file:
            yaml.safe_dump(data, file)

    def load_yaml_matrix(self) -> Iterator[Dict[str, Any]]:
        """
        Load a matrix from a YAML file and yield dictionaries representing all possible combinations of the matrix values.
        The YAML file should contain a "matrix" key with a dictionary of lists as its value. This method will generate all
        possible combinations of the values in the lists and yield them as dictionaries.
        Returns:
            Iterator[Dict[str, Any]]: An iterator over dictionaries, each representing a unique combination of the matrix values.
        Raises:
            FileNotFoundError: If the specified YAML file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """

        with open(self.file_path, "r") as file:
            original_dict = yaml.safe_load(file)
            matrix = original_dict.get("matrix", None)
        if matrix:
            original_dict.pop("matrix")
            file_text = yaml.dump(original_dict)
            keys, values = zip(*self.force_iterable(matrix).items())
            combinations = (dict(zip(keys, combo)) for combo in product(*values))
            for comb in combinations:
                yield yaml.safe_load(
                    re.sub(r"\$\{(\w+)\}", lambda m: str(comb[m.group(1)]), file_text)
                )
        else:
            yield self.load_yaml()

    @staticmethod
    def force_iterable(data: dict[Any, Any]) -> dict[Any, Any]:
        """
        Recursively converts all dictionary values to lists if they are not already lists.
        Args:
            data (dict): The dictionary to convert.
        Returns:
            dict: The dictionary with all values converted to lists.
        """
        for key, value in data.items():
            if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
                data[key] = [value]
        return data
