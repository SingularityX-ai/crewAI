"""Test Agent creation and execution basic functionality."""


from crewai.agent import Agent
from crewai.task import Task


def test_task_tool_reflect_agent_tools():
    """
    Test the reflection of tools in the task agent.

    This function tests the reflection of tools in the task agent by creating a fake tool and an agent, and then assigning the fake tool to the agent's tools. It then creates a task with a description and the agent, and asserts that the task's tools include the fake tool.

    Raises:
        AssertionError: If the task's tools do not include the fake tool.

    """

    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        """
        Fake tool

        Raises:
            No specific exceptions are raised.
        """

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[fake_tool],
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        agent=researcher,
    )

    assert task.tools == [fake_tool]


def test_task_tool_takes_precedence_ove_agent_tools():
    """
    Test that the task tool takes precedence over agent tools.

    This function sets up a test scenario where a task tool is created and assigned to a task. The test then asserts that the task tool takes precedence over the agent tools.

    Raises:
        AssertionError: If the task tools do not take precedence over the agent tools.

    """

    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        """
        Fake tool

        Raises:
            No exceptions are raised.

        Returns:
            None
        """

    @tool
    def fake_task_tool() -> None:
        """
        Fake tool

        This function does not raise any exceptions.

        Returns:
            None
        """

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[fake_tool],
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        agent=researcher,
        tools=[fake_task_tool],
        allow_delegation=False,
    )

    assert task.tools == [fake_task_tool]
