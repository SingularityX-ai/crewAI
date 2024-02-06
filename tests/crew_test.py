"""Test Agent creation and execution basic functionality."""

import json

import pytest

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler
from crewai.crew import Crew
from crewai.process import Process
from crewai.task import Task
from crewai.utilities import Logger, RPMController

ceo = Agent(
    role="CEO",
    goal="Make sure the writers in your company produce amazing content.",
    backstory="You're an long time CEO of a content creation agency with a Senior Writer on the team. You're now working on a new project and want to make sure the content produced is amazing.",
    allow_delegation=True,
)

researcher = Agent(
    role="Researcher",
    goal="Make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Write the best content about AI and AI agents.",
    backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
    allow_delegation=False,
)


def test_crew_config_conditional_requirement():
    """
    Test the conditional requirement for crew configuration.

    This function tests the conditional requirement for crew configuration by checking if the Crew class raises a ValueError when instantiated with a sequential process and a specific configuration.

    Raises:
        ValueError: If the Crew class instantiation with a sequential process and specific configuration does not raise a ValueError.

    """

    with pytest.raises(ValueError):
        Crew(process=Process.sequential)

    config = json.dumps(
        {
            "agents": [
                {
                    "role": "Senior Researcher",
                    "goal": "Make the best research and analysis on content about AI and AI agents",
                    "backstory": "You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
                },
                {
                    "role": "Senior Writer",
                    "goal": "Write the best content about AI and AI agents.",
                    "backstory": "You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
                },
            ],
            "tasks": [
                {
                    "description": "Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
                    "agent": "Senior Researcher",
                },
                {
                    "description": "Write a 1 amazing paragraph highlight for each idead that showcases how good an article about this topic could be, check references if necessary or search for more content but make sure it's unique, interesting and well written. Return the list of ideas with their paragraph and your notes.",
                    "agent": "Senior Writer",
                },
            ],
        }
    )
    parsed_config = json.loads(config)

    try:
        crew = Crew(process=Process.sequential, config=config)
    except ValueError:
        pytest.fail("Unexpected ValidationError raised")

    assert [agent.role for agent in crew.agents] == [
        agent["role"] for agent in parsed_config["agents"]
    ]
    assert [task.description for task in crew.tasks] == [
        task["description"] for task in parsed_config["tasks"]
    ]


def test_crew_config_with_wrong_keys():
    """
    Test for crew configuration with wrong keys.

    This function tests the crew configuration with wrong keys by creating JSON strings for no_tasks_config and no_agents_config. It then uses pytest to check if Crew raises a ValueError when provided with incorrect configurations.

    Raises:
        ValueError: If the crew configuration contains wrong keys.

    """

    no_tasks_config = json.dumps(
        {
            "agents": [
                {
                    "role": "Senior Researcher",
                    "goal": "Make the best research and analysis on content about AI and AI agents",
                    "backstory": "You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
                }
            ]
        }
    )

    no_agents_config = json.dumps(
        {
            "tasks": [
                {
                    "description": "Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
                    "agent": "Senior Researcher",
                }
            ]
        }
    )
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config='{"wrong_key": "wrong_value"}')
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config=no_tasks_config)
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config=no_agents_config)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_creation():
    """
    Test the creation of a crew with tasks and agents.

    This function creates a crew with a list of tasks and agents, and then kicks off the crew to perform the tasks. It asserts that the output matches the expected result.

    Raises:
        AssertionError: If the output of crew.kickoff() does not match the expected result.

    """

    tasks = [
        Task(
            description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
            agent=researcher,
        ),
        Task(
            description="Write a 1 amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
            agent=writer,
        ),
    ]

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    assert (
        crew.kickoff()
        == """1. **The Evolution of AI: From Old Concepts to New Frontiers** - Journey with us as we traverse the fascinating timeline of artificial intelligence - from its philosophical and mathematical infancy to the sophisticated, problem-solving tool it has become today. This riveting account will not only educate but also inspire, as we delve deep into the milestones that brought us here and shine a beacon on the potential that lies ahead.

2. **AI Agents in Healthcare: The Future of Medicine** - Imagine a world where illnesses are diagnosed before symptoms appear, where patient outcomes are not mere guesses but accurate predictions. This is the world AI is crafting in healthcare - a revolution that's saving lives and changing the face of medicine as we know it. This article will spotlight this transformative journey, underlining the profound impact AI is having on our health and well-being.

3. **AI and Ethics: Navigating the Moral Landscape of Artificial Intelligence** - As AI becomes an integral part of our lives, it brings along a plethora of ethical dilemmas. This thought-provoking piece will navigate the complex moral landscape of AI, addressing critical concerns like privacy, job displacement, and decision-making biases. It serves as a much-needed discussion platform for the societal implications of AI, urging us to look beyond the technology and into the mirror.

4. **Demystifying AI Algorithms: A Deep Dive into Machine Learning** - Ever wondered what goes on behind the scenes of AI? This enlightening article will break down the complex world of machine learning algorithms into digestible insights, unraveling the mystery of AI's 'black box'. It's a rare opportunity for the non-technical audience to appreciate the inner workings of AI, fostering a deeper understanding of this revolutionary technology.

5. **AI Startups: The Game Changers of the Tech Industry** - In the world of tech, AI startups are the bold pioneers charting new territories. This article will spotlight these game changers, showcasing how their innovative products and services are driving the AI revolution. It's a unique opportunity to catch a glimpse of the entrepreneurial side of AI, offering inspiration for the tech enthusiasts and dreamers alike."""
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_with_delegating_agents():
    """
    Test the crew with delegating agents.

    This function sets up a test scenario with a crew of agents and tasks, and then asserts the kickoff result.

    Raises:
        AssertionError: If the crew kickoff result does not match the expected value.

    """

    tasks = [
        Task(
            description="Produce and amazing 1 paragraph draft of an article about AI Agents.",
            agent=ceo,
        )
    ]

    crew = Crew(
        agents=[ceo, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    assert (
        crew.kickoff()
        == '"AI agents, the digital masterminds at the heart of the 21st-century revolution, are shaping a new era of intelligence and innovation. They are autonomous entities, capable of observing their environment, making decisions, and acting on them, all in pursuit of a specific goal. From streamlining operations in logistics to personalizing customer experiences in retail, AI agents are transforming how businesses operate. But their potential extends far beyond the corporate world. They are the sentinels protecting our digital frontiers, the virtual assistants making our lives easier, and the unseen hands guiding autonomous vehicles. As this technology evolves, AI agents will play an increasingly central role in our world, ushering in an era of unprecedented efficiency, personalization, and productivity. But with great power comes great responsibility, and understanding and harnessing this potential responsibly will be one of our greatest challenges and opportunities in the coming years."'
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_verbose_output(capsys):
    """
    Test the verbose output of the Crew class.

    Args:
    capsys: A built-in pytest fixture for capturing stdout and stderr.

    Raises:
    AssertionError: If the expected strings are not found in the captured output.

    Example:
    ```
    test_crew_verbose_output(capsys)
    ```
    """

    tasks = [
        Task(description="Research AI advancements.", agent=researcher),
        Task(description="Write about AI in healthcare.", agent=writer),
    ]

    crew = Crew(
        agents=[researcher, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = [
        "[DEBUG]: Working Agent: Researcher",
        "[INFO]: Starting Task: Research AI advancements.",
        "[DEBUG]: [Researcher] Task output:",
        "[DEBUG]: Working Agent: Senior Writer",
        "[INFO]: Starting Task: Write about AI in healthcare.",
        "[DEBUG]: [Senior Writer] Task output:",
    ]

    for expected_string in expected_strings:
        assert expected_string in captured.out

    # Now test with verbose set to False
    crew._logger = Logger(verbose_level=False)
    crew.kickoff()
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_verbose_levels_output(capsys):
    """
    Test the verbose levels output of the Crew class.

    Args:
    capsys: A built-in pytest fixture for capturing stdout and stderr.

    Raises:
    AssertionError: If the expected strings are not found in the captured output.
    """

    tasks = [Task(description="Write about AI advancements.", agent=researcher)]

    crew = Crew(agents=[researcher], tasks=tasks, process=Process.sequential, verbose=1)

    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = ["Working Agent: Researcher", "[Researcher] Task output:"]

    for expected_string in expected_strings:
        assert expected_string in captured.out

    # Now test with verbose set to 2
    crew._logger = Logger(verbose_level=2)
    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = [
        "Working Agent: Researcher",
        "Starting Task: Write about AI advancements.",
        "[Researcher] Task output:",
    ]

    for expected_string in expected_strings:
        assert expected_string in captured.out


@pytest.mark.vcr(filter_headers=["authorization"])
def test_cache_hitting_between_agents():
    """    Test cache hitting between agents.

        This function tests the cache hitting between agents by creating a
        multiplier tool and using it to perform multiplication tasks for different
        agents. It then checks if the cache is being utilized correctly.

        Raises:
            AssertionError: If the cache is not being utilized correctly.

    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def multiplier(numbers) -> float:
        """
        Useful for when you need to multiply two numbers together.

        Args:
        - numbers (str): A comma separated list of numbers of length two, representing the two numbers you want to multiply together.

        Returns:
        - float: The result of multiplying the two input numbers together.

        Raises:
        - ValueError: If the input format is incorrect or if the input numbers are not valid integers.
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    tasks = [
        Task(
            description="What is 2 tims 6? Return only the number.",
            tools=[multiplier],
            agent=ceo,
        ),
        Task(
            description="What is 2 times 6? Return only the number.",
            tools=[multiplier],
            agent=researcher,
        ),
    ]

    crew = Crew(
        agents=[ceo, researcher],
        tasks=tasks,
    )

    assert crew._cache_handler._cache == {}
    output = crew.kickoff()
    assert crew._cache_handler._cache == {"multiplier-2,6": "12"}
    assert output == "12"

    with patch.object(CacheHandler, "read") as read:
        read.return_value = "12"
        crew.kickoff()
        read.assert_called_with("multiplier", "2,6")


@pytest.mark.vcr(filter_headers=["authorization"])
def test_api_calls_throttling(capsys):
    """    Test the throttling of API calls.

        This function tests the throttling of API calls by simulating the usage of a tool to get the final answer without actually giving it. It creates an agent, a task, and a crew to carry out the test, and then uses a mock object to control the RPM (Revolutions Per Minute) and verify that the maximum RPM is reached.

        Args:
            capsys: A built-in pytest fixture for capturing stdout and stderr.

        Raises:
            AssertionError: If the maximum RPM is not reached or if the expected message is not found in the captured output.

    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """
        Get the final answer but don't give it yet, just re-use this tool non-stop.

        Args:
        - numbers: Input numbers

        Returns:
        - float: The final answer

        Raises:
        - None
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=5,
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description="Don't give a Final Answer, instead keep using the `get_final_answer` tool.",
        tools=[get_final_answer],
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], max_rpm=2, verbose=2)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called()
