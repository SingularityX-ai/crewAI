"""Test Agent creation and execution basic functionality."""

import pytest

from crewai.agent import Agent
from crewai.tools.agent_tools import AgentTools

researcher = Agent(
    role="researcher",
    goal="make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology",
    allow_delegation=False,
)
tools = AgentTools(agents=[researcher])


@pytest.mark.vcr(filter_headers=["authorization"])
def test_delegate_work():
    """
    Test the delegate_work function.

    This function tests the delegate_work function from the tools module by providing a specific command and checking if the result matches the expected output.

    Raises:
        Any exceptions that may be raised during the execution of the delegate_work function.

    """

    result = tools.delegate_work(
        command="researcher|share your take on AI Agents|I heard you hate them"
    )

    assert (
        result
        == "I apologize if my previous statements have given you the impression that I hate AI agents. As a technology researcher, I don't hold personal sentiments towards AI or any other technology. Rather, I analyze them objectively based on their capabilities, applications, and implications. AI agents, in particular, are a fascinating domain of research. They hold tremendous potential in automating and optimizing various tasks across industries. However, like any other technology, they come with their own set of challenges, such as ethical considerations around privacy and decision-making. My objective is to understand these technologies in depth and provide a balanced view."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question():
    """
    Test the ask_question function.

    This function tests the ask_question function by providing a specific command and checking if the result matches the expected response.

    Raises:
        AssertionError: If the result of the ask_question function does not match the expected response.

    """

    result = tools.ask_question(
        command="researcher|do you hate AI Agents?|I heard you LOVE them"
    )

    assert (
        result
        == "As an AI, I don't possess feelings or emotions, so I don't love or hate anything. However, I can provide detailed analysis and research on AI agents. They are a fascinating field of study with the potential to revolutionize many industries, although they also present certain challenges and ethical considerations."
    )


def test_can_not_self_delegate():
    """
    Test that self delegation is not allowed.

    Raises:
        NotImplementedError: If self delegation is attempted.
    """

    # TODO: Add test for self delegation
    pass


def test_delegate_work_with_wrong_input():
    """
    Test if the delegate work with wrong input.

    This function tests the behavior of the delegate when provided with wrong input. It calls the `ask_question`
    function from the `tools` module with a specific command and asserts that the result matches an expected error message.

    Raises:
        AssertionError: If the result does not match the expected error message.

    """

    result = tools.ask_question(command="writer|share your take on AI Agents")

    assert (
        result
        == "\nError executing tool. Missing exact 3 pipe (|) separated values. For example, `coworker|task|context`. I need to make sure to pass context as context.\n"
    )


def test_delegate_work_to_wrong_agent():
    """
    Test for delegating work to the wrong agent.

    This function tests the behavior of delegating work to the wrong agent by simulating the process and asserting the expected result.

    Raises:
        <Exception Type>: <Description of when this exception might be raised>

    Returns:
        None
    """

    result = tools.ask_question(
        command="writer|share your take on AI Agents|I heard you hate them"
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: researcher.\n"
    )


def test_ask_question_to_wrong_agent():
    """
    Test for asking a question to the wrong agent.

    This test checks the behavior of the ask_question function when the command is
    directed to a wrong agent. It verifies that the function returns an error message
    indicating that the specified agent is not found.

    Raises:
        AssertionError: If the result returned by the ask_question function does not match
        the expected error message.
    """

    result = tools.ask_question(
        command="writer|do you hate AI Agents?|I heard you LOVE them"
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: researcher.\n"
    )
