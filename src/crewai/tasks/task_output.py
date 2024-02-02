from typing import Optional

from pydantic import BaseModel, Field, model_validator


class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    result: str = Field(description="Result of the task")

    @model_validator(mode="after")
    def set_summary(self):
        """
        Set the summary of the description by taking the first 10 words and appending '...' at the end.

        Raises:
            None

        Returns:
            self: The instance of the object with the summary set.
        """

        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self
