from typing import List, Optional

from langchain.tools import Tool
from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.utilities import I18N


class AgentTools(BaseModel):
    """Default tools around agent delegation"""

    agents: List[Agent] = Field(description="List of agents in this crew.")
    i18n: Optional[I18N] = Field(
        default=I18N(), description="Internationalization settings."
    )

    def tools(self):
        """
        Return a list of tools for delegating work and asking questions to co-workers.

        This method returns a list of Tool objects, each representing a specific tool for delegating work or asking questions
        to co-workers.

        Returns:
            list: A list of Tool objects representing different tools for delegating work and asking questions.

        Raises:
            SomeException: This exception is raised if there is a specific condition under which the method cannot proceed.

        """


        return [
            Tool.from_function(
                func=self.delegate_work,
                name="Delegate work to co-worker",
                description=self.i18n.tools("delegate_work").format(
                    coworkers=", ".join([agent.role for agent in self.agents])
                ),
            ),
            Tool.from_function(
                func=self.ask_question,
                name="Ask question to co-worker",
                description=self.i18n.tools("ask_question").format(
                    coworkers=", ".join([agent.role for agent in self.agents])
                ),
            ),
        ]

    def delegate_work(self, command):

        """
        Useful to delegate a specific task to a coworker.

        Args:
        - command (str): The command to be executed.

        Raises:
        - (Exception): If the command execution fails.

        Returns:
        - The result of executing the command.
        """
        return self.__execute(command)

    def ask_question(self, command):

        """
        Useful to ask a question, opinion or take from a coworker.

        Args:
        - command (str): The command to be executed.

        Returns:
        - The result of executing the command.

        Raises:
        - Any exceptions raised by the execution of the command.
        """
        return self.__execute(command)

    def __execute(self, command):
        
        """
        Execute the command.

        Args:
        - command (str): A string representing the command in the format "agent|task|context".

        Returns:
        - str: The result of executing the task by the specified agent.

        Raises:
        - ValueError: If the command does not contain all three parts (agent, task, context).
        - ValueError: If any of the parts (agent, task, context) is empty.

        """
        try:
            agent, task, context = command.split("|")
        except ValueError:
            return self.i18n.errors("agent_tool_missing_param")

        if not agent or not task or not context:
            return self.i18n.errors("agent_tool_missing_param")

        agent = [
            available_agent
            for available_agent in self.agents
            if available_agent.role == agent
        ]

        if not agent:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers=", ".join([agent.role for agent in self.agents])
            )

        agent = agent[0]
        return agent.execute_task(task, context)
