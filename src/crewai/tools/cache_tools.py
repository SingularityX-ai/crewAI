from langchain.tools import Tool
from pydantic import BaseModel, ConfigDict, Field

from crewai.agents.cache import CacheHandler


class CacheTools(BaseModel):
    """Default tools to hit the cache."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(
        description="Cache Handler for the crew",
        default=CacheHandler(),
    )

    def tool(self):
        """
        Return a Tool object created from the hit_cache function.

        Args:
            self: The instance of the class.

        Returns:
            Tool: A Tool object.

        Raises:
            None
        """


        return Tool.from_function(
            func=self.hit_cache,
            name=self.name,
            description="Reads directly from the cache",
        )

    def hit_cache(self, key):
        """
        Retrieve data from the cache for the given key.

        Args:
        key (str): The key used to retrieve data from the cache. It should be in the format "tool:<tool_name>|input:<input_data>".

        Returns:
        The data retrieved from the cache for the given key.

        Raises:
        KeyError: If the specified key is not found in the cache.
        """

        
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)
