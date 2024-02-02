"""Test Agent creation and execution basic functionality."""

from unittest.mock import patch

import pytest
from langchain.tools import tool
from langchain_openai import ChatOpenAI as OpenAI

from crewai import Agent, Crew, Task
from crewai.agents.cache import CacheHandler
from crewai.agents.executor import CrewAgentExecutor
from crewai.utilities import RPMController


def test_agent_creation():
    """
    Test the creation of an Agent instance.

    Raises:
        AssertionError: If the role, goal, or backstory attributes are not set correctly.

    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.tools == []


def test_agent_default_values():
    """
    Test the default values of the Agent class.

    Raises:
        AssertionError: If any of the default values are not as expected.

    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0.7
    assert agent.llm.verbose == False
    assert agent.allow_delegation == True


def test_custom_llm():
    """
    Test the custom LLM (Language Model) configuration for the Agent class.

    This function creates an Agent instance with a custom LLM configuration and asserts its properties.

    Raises:
        AssertionError: If the LLM is not an instance of OpenAI or if its model_name and temperature properties are not as expected.

    """

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_without_memory():
    """
    Test the Agent class without memory.

    This function creates an Agent instance with memory set to False and another instance with memory set to True.
    It then executes a task and asserts the results and memory state of the agents.

    Raises:
        AssertionError: If the test results or memory state assertions fail.
    """

    no_memory_agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        memory=False,
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    memory_agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        memory=True,
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    result = no_memory_agent.execute_task("How much is 1 + 1?")

    assert result == "1 + 1 equals 2."
    assert no_memory_agent.agent_executor.memory is None
    assert memory_agent.agent_executor.memory is not None


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution():
    """
    Test the execution of the Agent class.

    This function creates an instance of the Agent class with the specified role, goal, backstory, and delegation settings.
    It then executes a task and asserts that the output matches the expected result.

    Raises:
        AssertionError: If the output of the executed task does not match the expected result.

    """

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    output = agent.execute_task("How much is 1 + 1?")
    assert output == "2"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution_with_tools():
    """
    Test the execution of an agent with tools.

    This function tests the execution of an agent with tools. It defines a multiplier tool
    that multiplies two numbers together. The agent is then created with a specified role,
    goal, backstory, and tools, and the task "What is 3 times 4" is executed. The expected
    output is "12".

    Raises:
        AssertionError: If the output of the executed task does not match the expected output.

    """

    @tool
    def multiplier(numbers) -> float:
        """
        Useful for when you need to multiply two numbers together.

        Args:
        numbers (str): A comma separated list of numbers of length two, representing the two numbers you want to multiply together.

        Returns:
        float: The result of multiplying the two input numbers together.

        Raises:
        ValueError: If the input format is incorrect or if the input numbers are not valid.

        Example:
        >>> multiplier("2,3")
        6
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
    )

    output = agent.execute_task("What is 3 times 4")
    assert output == "12"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_logging_tool_usage():
    """    Test the usage of logging tools.

        This function tests the usage of logging tools by creating a multiplier tool and an agent, and then executing a task to check the output and tool usage.

        Raises:
            AssertionError: If the output or tool usage does not match the expected values.

    """

    @tool
    def multiplier(numbers) -> float:
        """
        Useful for when you need to multiply two numbers together.

        Args:
        numbers (str): A comma separated list of numbers of length two, representing the two numbers you want to multiply together.

        Returns:
        float: The result of multiplying the two input numbers together.

        Raises:
        ValueError: If the input is not in the correct format (e.g., not a comma separated list of two numbers).
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
        verbose=True,
    )

    assert agent.tools_handler.last_used_tool == {}
    output = agent.execute_task("What is 3 times 5?")
    tool_usage = {
        "tool": "multiplier",
        "input": "3,5",
    }

    assert output == "3 times 5 is 15."
    assert agent.tools_handler.last_used_tool == tool_usage


@pytest.mark.vcr(filter_headers=["authorization"])
def test_cache_hitting():
    """
    Test the cache handling functionality of the Agent class.

    This test case checks if the Agent class correctly handles caching of tool results.

    Raises:
        AssertionError: If the cache handling functionality does not work as expected.
    """

    @tool
    def multiplier(numbers) -> float:
        """        Multiply two numbers together.

            Args:
                numbers (str): A comma separated list of two numbers to be multiplied.

            Returns:
                float: The result of multiplying the two input numbers together.

            Raises:
                ValueError: If the input does not contain exactly two numbers separated by a comma.

            Example:
                >>> multiplier('2,3')
                6
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    cache_handler = CacheHandler()

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
        cache_handler=cache_handler,
        verbose=True,
    )

    output = agent.execute_task("What is 2 times 6 times 3?")
    output = agent.execute_task("What is 3 times 3?")
    assert cache_handler._cache == {
        "multiplier-12,3": "36",
        "multiplier-2,6": "12",
        "multiplier-3,3": "9",
    }

    output = agent.execute_task("What is 2 times 6 times 3? Return only the number")
    assert output == "36"

    with patch.object(CacheHandler, "read") as read:
        read.return_value = "0"
        output = agent.execute_task("What is 2 times 6?")
        assert output == "0"
        read.assert_called_with("multiplier", "2,6")


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution_with_specific_tools():
    """
    Test the execution of an agent with specific tools.

    This function defines a test scenario for executing an agent with specific tools. It creates a multiplier tool
    using the `@tool` decorator and then creates an agent with specified role, goal, backstory, and delegation settings.
    The agent is then tasked with a question and the output is checked against the expected result.

    Raises:
        AssertionError: If the output does not match the expected result.

    """

    @tool
    def multiplier(numbers) -> float:
        """
        Useful for when you need to multiply two numbers together.

        Args:
        numbers (str): A comma separated list of numbers of length two, representing the two numbers you want to multiply together.

        Returns:
        float: The result of multiplying the two input numbers together.

        Raises:
        ValueError: If the input format is incorrect or if the input numbers are not valid.

        Example:
        >>> multiplier("2,3")
        6
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    output = agent.execute_task(task="What is 3 times 4", tools=[multiplier])
    assert output == "3 times 4 is 12."


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_custom_max_iterations():
    """
    Test the custom max iterations for the agent.

    This function tests the custom max iterations for the agent by creating a test agent with specific attributes and executing a task using a mocked method.

    Raises:
        Any exceptions that may occur during the execution of the test.

    """

    @tool
    def get_final_answer(numbers) -> float:
        """
        Get the final answer but don't give it yet, just re-use this
        tool non-stop.

        :param numbers: Input numbers
        :type numbers: Any
        :return: The final answer
        :rtype: float
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=1,
        allow_delegation=False,
    )

    with patch.object(
        CrewAgentExecutor, "_iter_next_step", wraps=agent.agent_executor._iter_next_step
    ) as private_mock:
        agent.execute_task(
            task="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            tools=[get_final_answer],
        )
        private_mock.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_moved_on_after_max_iterations():
    """
    Test that the agent moves on after reaching the maximum iterations.

    This test function checks whether the agent moves on after reaching the maximum iterations.

    Raises:
        AssertionError: If the agent does not move on after reaching the maximum iterations.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """
        Get the final answer but don't give it yet, just re-use this
        tool non-stop.

        Args:
        - numbers: A list of numbers.

        Returns:
        - float: The final answer.

        Raises:
        - This function does not raise any exceptions.
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=3,
        allow_delegation=False,
    )

    with patch.object(
        CrewAgentExecutor, "_force_answer", wraps=agent.agent_executor._force_answer
    ) as private_mock:
        output = agent.execute_task(
            task="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            tools=[get_final_answer],
        )
        assert (
            output
            == "I have used the tool multiple times and the final answer remains 42."
        )
        private_mock.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_respect_the_max_rpm_set(capsys):
    """
    Test if the agent respects the maximum RPM set.

    Args:
    - capsys: A built-in pytest fixture to capture stdout and stderr.

    Raises:
    No specific exceptions are raised.

    Returns:
    No return value.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """
        Get the final answer but don't give it yet, just re-use this tool non-stop.

        Args:
        - numbers: A list of numbers for calculation

        Returns:
        - float: The final answer calculated from the input numbers

        Raises:
        - This function does not raise any specific exceptions.
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=5,
        max_rpm=1,
        verbose=True,
        allow_delegation=False,
    )

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        output = agent.execute_task(
            task="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            tools=[get_final_answer],
        )
        assert (
            output
            == "I've used the `get_final_answer` tool multiple times and it consistently returns the number 42."
        )
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_respect_the_max_rpm_set_over_crew_rpm(capsys):
    """    Test if the agent respects the maximum RPM set over the crew RPM.

        Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr.

        Raises:
        AssertionError: If the expected output does not match the actual output.

    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """
        Get the final answer but don't give it yet, just re-use this
        tool non-stop.

        Args:
        - numbers: A list of numbers.

        Returns:
        - float: The final answer.

        Raises:
        - This function does not raise any exceptions.
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=4,
        max_rpm=10,
        verbose=True,
    )

    task = Task(
        description="Don't give a Final Answer, instead keep using the `get_final_answer` tool.",
        tools=[get_final_answer],
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], max_rpm=1, verbose=2)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." not in captured.out
        moveon.assert_not_called()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_without_max_rpm_respet_crew_rpm(capsys):
    """    Test the behavior of an agent when the maximum RPM is not respected by the crew RPM.

        Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr output.

        Raises:
        AssertionError: If the expected output does not match the actual output.

    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """
        Get the final answer but don't give it yet, just re-use this
        tool non-stop.

        Args:
        - numbers: A list of numbers.

        Returns:
        - float: The final answer.

        Raises:
        - This function does not raise any exceptions.
        """
        return 42

    agent1 = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_rpm=10,
        verbose=True,
    )

    agent2 = Agent(
        role="test role2",
        goal="test goal2",
        backstory="test backstory2",
        max_iter=2,
        verbose=True,
    )

    tasks = [
        Task(
            description="Just say hi.",
            agent=agent1,
        ),
        Task(
            description="Don't give a Final Answer, instead keep using the `get_final_answer` tool.",
            tools=[get_final_answer],
            agent=agent2,
        ),
    ]

    crew = Crew(agents=[agent1, agent2], tasks=tasks, max_rpm=1, verbose=2)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Action: get_final_answer" in captured.out
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called_once()
