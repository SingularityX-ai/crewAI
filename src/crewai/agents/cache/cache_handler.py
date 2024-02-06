from typing import Optional

from pydantic import PrivateAttr


class CacheHandler:
    """Callback handler for tool usage."""

    _cache: PrivateAttr = {}

    def __init__(self):
        """
        Initialize the object.

        Args:
            self: The object itself.

        Returns:
            None

        Raises:
            None
        """

        self._cache = {}

    def add(self, tool, input, output):
        """
        Add the output of a tool to the cache.

        Args:
        tool (str): The name of the tool.
        input (str): The input string.
        output (str): The output string.

        Raises:
        No exceptions are explicitly raised.

        Returns:
        None
        """


        input = input.strip()
        self._cache[f"{tool}-{input}"] = output

    def read(self, tool, input) -> Optional[str]:
        """
        Read the cached data for the given tool and input.

        Args:
            tool: The tool for which the data is being read.
            input: The input for which the data is being read.

        Returns:
            Optional[str]: The cached data for the given tool and input, if available.

        Raises:
            None
        """

        
        input = input.strip()
        return self._cache.get(f"{tool}-{input}")
