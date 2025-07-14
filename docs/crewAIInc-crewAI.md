html
<!DOCTYPE html>
<html>
<head>
<title>CrewAI: Build Autonomous AI Agents with Speed and Control</title>
<meta name="description" content="CrewAI is a cutting-edge Python framework for building autonomous AI agents. Experience the power of Crews and Flows for unparalleled control and flexibility. Get started today!">
<meta name="keywords" content="AI agents, multi-agent, automation, Python, framework, Crews, Flows, LLM, autonomous agents, CrewAI">
<style>
body {
    font-family: sans-serif;
    line-height: 1.6;
    margin: 20px;
}
h1, h2, h3 {
    color: #333;
}
a {
    color: #007BFF;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
.key-features ul {
    list-style-type: disc;
    padding-left: 20px;
}
.logo {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 60%; /* Adjust as needed */
}
.shields {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 10px;
}
.badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 10px;
}

</style>
</head>
<body>

<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" alt="CrewAI Logo" class="logo">
  </a>
</p>

<div class="shields">
  <a href="https://trendshift.io/repositories/11239" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/11239" alt="crewAIInc%2FcrewAI | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

<div align="center">
    <a href="https://crewai.com">Homepage</a> |
    <a href="https://docs.crewai.com">Docs</a> |
    <a href="https://app.crewai.com">Start Cloud Trial</a> |
    <a href="https://blog.crewai.com">Blog</a> |
    <a href="https://community.crewai.com">Forum</a>
</div>

<div class="shields">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="https://img.shields.io/github/stars/crewAIInc/crewAI" alt="GitHub Repo stars">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/network/members">
    <img src="https://img.shields.io/github/forks/crewAIInc/crewAI" alt="GitHub forks">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/issues">
    <img src="https://img.shields.io/github/issues/crewAIInc/crewAI" alt="GitHub issues">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/pulls">
    <img src="https://img.shields.io/github/issues-pr/crewAIInc/crewAI" alt="GitHub pull requests">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
</div>

<div class="badges">
  <a href="https://pypi.org/project/crewai/">
    <img src="https://img.shields.io/pypi/v/crewai" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/crewai/">
    <img src="https://img.shields.io/pypi/dm/crewai" alt="PyPI downloads">
  </a>
  <a href="https://twitter.com/crewAIInc">
    <img src="https://img.shields.io/twitter/follow/crewAIInc?style=social" alt="Twitter Follow">
  </a>
</div>


<h1>CrewAI: Unleash the Power of Autonomous AI Agents for Unprecedented Automation</h1>
<p><b>CrewAI</b> is a lightning-fast Python framework for building autonomous AI agents, offering unparalleled flexibility and control for complex automation tasks.</p>

<h2>Key Features</h2>
<div class="key-features">
<ul>
    <li><b>Standalone &amp; Lean:</b> Built from scratch, independent of other frameworks like LangChain, ensuring faster execution and reduced resource consumption.</li>
    <li><b>Crews:</b>  Orchestrate teams of AI agents with true autonomy and collaborative intelligence, for optimal teamwork.</li>
    <li><b>Flows:</b> Implement granular, event-driven control for precise task orchestration and seamless integration with production code.</li>
    <li><b>Deep Customization:</b>  Tailor workflows, agent behaviors, and internal prompts to fit any use case.</li>
    <li><b>High Performance:</b>  Optimized for speed, delivering results efficiently across both simple and complex tasks.</li>
    <li><b>Thriving Community:</b> Supported by extensive documentation and a growing community of developers for comprehensive assistance.</li>
</ul>
</div>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#why-crewai">Why CrewAI?</a></li>
  <li><a href="#getting-started">Getting Started</a></li>
  <li><a href="#key-features">Key Features</a></li>
  <li><a href="#examples">Examples</a></li>
  <li><a href="#connecting-your-crew-to-a-model">Connecting Your Crew to a Model</a></li>
  <li><a href="#how-crewai-compares">How CrewAI Compares</a></li>
  <li><a href="#contribution">Contribution</a></li>
  <li><a href="#telemetry">Telemetry</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#frequently-asked-questions-faq">Frequently Asked Questions (FAQ)</a></li>
</ul>

<h2 id="why-crewai">Why CrewAI?</h2>
<p><img src="docs/images/asset.png" alt="CrewAI Logo" width="100%" style="margin-bottom: 30px;"></p>
<p>CrewAI delivers the best-in-class combination of speed, flexibility, and control with either Crews of AI Agents or Flows of Events.</p>
<ul>
    <li><b>Standalone Framework:</b> Independent of LangChain or other agent frameworks.</li>
    <li><b>High Performance:</b> Optimized for speed and minimal resource usage, enabling faster execution.</li>
    <li><b>Flexible Low Level Customization:</b> Complete freedom to customize at both high and low levels - from overall workflows and system architecture to granular agent behaviors, internal prompts, and execution logic.</li>
    <li><b>Ideal for Every Use Case:</b> Proven effective for both simple tasks and highly complex, real-world, enterprise-grade scenarios.</li>
    <li><b>Robust Community:</b> Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.</li>
</ul>
<p>Empower developers and enterprises to confidently build intelligent automations, bridging the gap between simplicity, flexibility, and performance.</p>

<h2 id="getting-started">Getting Started</h2>
<p>Setup and run your first CrewAI agents by following this tutorial.</p>

<p><a href="https://www.youtube.com/watch?v=-kSOTtYzgEw"><img src="https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg" alt="CrewAI Getting Started Tutorial"></a></p>

<p>Learn CrewAI through our comprehensive courses:</p>
<ul>
  <li><a href="https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/">Multi AI Agent Systems with CrewAI</a> - Master the fundamentals of multi-agent systems</li>
  <li><a href="https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/">Practical Multi AI Agents and Advanced Use Cases with CrewAI</a> - Deep dive into advanced implementations</li>
</ul>

<h3>Installation</h3>
<p>Ensure you have Python >=3.10 <3.14 installed on your system. CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.</p>
<p>First, install CrewAI:</p>
<pre><code>pip install crewai</code></pre>

<p>If you want to install the 'crewai' package along with its optional features that include additional tools for agents, you can do so by using the following command:</p>
<pre><code>pip install 'crewai[tools]'</code></pre>

<p>The command above installs the basic package and also adds extra components which require more dependencies to function.</p>

<h3>Troubleshooting Dependencies</h3>
<p>If you encounter issues during installation or usage, here are some common solutions:</p>

<h4>Common Issues</h4>
<ol>
    <li><b>ModuleNotFoundError: No module named 'tiktoken'</b>
        <ul>
            <li>Install tiktoken explicitly: <code>pip install 'crewai[embeddings]'</code></li>
            <li>If using embedchain or other tools: <code>pip install 'crewai[tools]'</code></li>
        </ul>
    </li>
    <li><b>Failed building wheel for tiktoken</b>
        <ul>
            <li>Ensure Rust compiler is installed (see installation steps above)</li>
            <li>For Windows: Verify Visual C++ Build Tools are installed</li>
            <li>Try upgrading pip: <code>pip install --upgrade pip</code></li>
            <li>If issues persist, use a pre-built wheel: <code>pip install tiktoken --prefer-binary</code></li>
        </ul>
    </li>
</ol>

<h3>Creating a CrewAI project</h3>
<p>To create a new CrewAI project, run the following CLI (Command Line Interface) command:</p>
<pre><code>crewai create crew &lt;project_name&gt;</code></pre>

<p>This command creates a new project folder with the following structure:</p>
<pre><code>my_project/
├── .gitignore
├── pyproject.toml
├── README.md
├── .env
└── src/
    └── my_project/
        ├── __init__.py
        ├── main.py
        ├── crew.py
        ├── tools/
        │   ├── custom_tool.py
        │   └── __init__.py
        └── config/
            ├── agents.yaml
            └── tasks.yaml
</code></pre>

<p>You can now start developing your crew by editing the files in the <code>src/my_project</code> folder. The <code>main.py</code> file is the entry point of the project, the <code>crew.py</code> file is where you define your crew, the <code>agents.yaml</code> file is where you define your agents, and the <code>tasks.yaml</code> file is where you define your tasks.</p>

<h4>To customize your project, you can:</h4>
<ul>
    <li>Modify <code>src/my_project/config/agents.yaml</code> to define your agents.</li>
    <li>Modify <code>src/my_project/config/tasks.yaml</code> to define your tasks.</li>
    <li>Modify <code>src/my_project/crew.py</code> to add your own logic, tools, and specific arguments.</li>
    <li>Modify <code>src/my_project/main.py</code> to add custom inputs for your agents and tasks.</li>
    <li>Add your environment variables into the <code>.env</code> file.</li>
</ul>

<h4>Example of a simple crew with a sequential process:</h4>

Instantiate your crew:

<pre><code>crewai create crew latest-ai-development</code></pre>

Modify the files as needed to fit your use case:

<h5>agents.yaml</h5>
<pre><code># src/my_project/config/agents.yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.

reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} data analysis and research findings
  backstory: >
    You're a meticulous analyst with a keen eye for detail. You're known for
    your ability to turn complex data into clear and concise reports, making
    it easy for others to understand and act on the information you provide.
</code></pre>

<h5>tasks.yaml</h5>
<pre><code># src/my_project/config/tasks.yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information given
    the current year is 2025.
  expected_output: >
    A list with 10 bullet points of the most relevant information about {topic}
  agent: researcher

reporting_task:
  description: >
    Review the context you got and expand each topic into a full section for a report.
    Make sure the report is detailed and contains any and all relevant information.
  expected_output: >
    A fully fledge reports with the mains topics, each with a full section of information.
    Formatted as markdown without '```'
  agent: reporting_analyst
  output_file: report.md
</code></pre>

<h5>crew.py</h5>
<pre><code># src/my_project/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class LatestAiDevelopmentCrew():
	"""LatestAiDevelopment crew"""
	agents: List[BaseAgent]
	tasks: List[Task]

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			tools=[SerperDevTool()]
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the LatestAiDevelopment crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
</code></pre>

<h5>main.py</h5>
<pre><code>#!/usr/bin/env python
# src/my_project/main.py
import sys
from latest_ai_development.crew import LatestAiDevelopmentCrew

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI Agents'
    }
    LatestAiDevelopmentCrew().crew().kickoff(inputs=inputs)
</code></pre>

<h3>Running Your Crew</h3>
<p>Before running your crew, make sure you have the following keys set as environment variables in your <code>.env</code> file:</p>
<ul>
    <li>An <a href="https://platform.openai.com/account/api-keys">OpenAI API key</a> (or other LLM API key): <code>OPENAI_API_KEY=sk-...</code></li>
    <li>A <a href="https://serper.dev/">Serper.dev</a> API key: <code>SERPER_API_KEY=YOUR_KEY_HERE</code></li>
</ul>

<p>Lock the dependencies and install them by using the CLI command but first, navigate to your project directory:</p>
<pre><code>cd my_project
crewai install (Optional)
</code></pre>

<p>To run your crew, execute the following command in the root of your project:</p>
<pre><code>crewai run</code></pre>

<p>or</p>
<pre><code>python src/my_project/main.py</code></pre>

<p>If an error happens due to the usage of poetry, please run the following command to update your crewai package:</p>
<pre><code>crewai update</code></pre>

<p>You should see the output in the console and the <code>report.md</code> file should be created in the root of your project with the full final report.</p>

<p>In addition to the sequential process, you can use the hierarchical process, which automatically assigns a manager to the defined crew to properly coordinate the planning and execution of tasks through delegation and validation of results. <a href="https://docs.crewai.com/core-concepts/Processes/">See more about the processes here</a>.</p>

<h2 id="examples">Examples</h2>

<p>You can test different real life examples of AI crews in the <a href="https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file">CrewAI-examples repo</a>:</p>
<ul>
    <li><a href="https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator">Landing Page Generator</a></li>
    <li><a href="https://docs.crewai.com/how-to/Human-Input-on-Execution">Having Human input on the execution</a></li>
    <li><a href="https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner">Trip Planner</a></li>
    <li><a href="https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis">Stock Analysis</a></li>
</ul>

<h3>Quick Tutorial</h3>
<p><a href="https://www.youtube.com/watch?v=tnejrr-0a94"><img src="https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg" alt="CrewAI Tutorial"></a></p>

<h3>Write Job Descriptions</h3>
<p><a href="https://github.com/crewAIInc/crewAI-examples/tree/main/job-posting">Check out code for this example</a> or watch a video below:</p>
<p><a href="https://www.youtube.com/watch?v=u98wEMz-9to"><img src="https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg" alt="Jobs postings"></a></p>

<h3>Trip Planner</h3>
<p><a href="https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner">Check out code for this example</a> or watch a video below:</p>
<p><a href="https://www.youtube.com/watch?v=xis7rWp-hjs"><img src="https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg" alt="Trip Planner"></a></p>

<h3>Stock Analysis</h3>
<p><a href="https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis">Check out code for this example</a> or watch a video below:</p>
<p><a href="https://www.youtube.com/watch?v=e0Uj4yWdaAg"><img src="https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg" alt="Stock Analysis"></a></p>

<h3>Using Crews and Flows Together</h3>

<p>CrewAI's power truly shines when combining Crews with Flows to create sophisticated automation pipelines.</p>
<p>CrewAI flows support logical operators like <code>or_</code> and <code>and_</code> to combine multiple conditions. This can be used with <code>@start</code>, <code>@listen</code>, or <code>@router</code> decorators to create complex triggering conditions.</p>
<ul>
  <li><code>or_</code>: Triggers when any of the specified conditions are met.</li>
  <li><code>and_</code>: Triggers when all of the specified conditions are met.</li>
</ul>

<p>Here's how you can orchestrate multiple Crews within a Flow:</p>
<pre><code>from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import Crew, Agent, Task, Process
from pydantic import BaseModel

# Define structured state for precise control
class MarketState(BaseModel):
    sentiment: str = "neutral"
    confidence: float = 0.0
    recommendations: list = []

class AdvancedAnalysisFlow(Flow[MarketState]):
    @start()
    def fetch_market_data(self):
        # Demonstrate low-level control with structured state
        self.state.sentiment = "analyzing"
        return {"sector": "tech", "timeframe": "1W"}  # These parameters match the task description template

    @listen(fetch_market_data)
    def analyze_with_crew(self, market_data):
        # Show crew agency through specialized roles
        analyst = Agent(
            role="Senior Market Analyst",
            goal="Conduct deep market analysis with expert insight",
            backstory="You're a veteran analyst known for identifying subtle market patterns"
        )
        researcher = Agent(
            role="Data Researcher",
            goal="Gather and validate supporting market data",
            backstory="You excel at finding and correlating multiple data sources"
        )

        analysis_task = Task(
            description="Analyze {sector} sector data for the past {timeframe}",
            expected_output="Detailed market analysis with confidence score",
            agent=analyst
        )
        research_task = Task(
            description="Find supporting data to validate the analysis",
            expected_output="Corroborating evidence and potential contradictions",
            agent=researcher
        )

        # Demonstrate crew autonomy
        analysis_crew = Crew(
            agents=[analyst, researcher],
            tasks=[analysis_task, research_task],
            process=Process.sequential,
            verbose=True
        )
        return analysis_crew.kickoff(inputs=market_data)  # Pass market_data as named inputs

    @router(analyze_with_crew)
    def determine_next_steps(self):
        # Show flow control with conditional routing
        if self.state.confidence > 0.8:
            return "high_confidence"
        elif self.state.confidence > 0.5:
            return "medium_confidence"
        return "low_confidence"

    @listen("high_confidence")
    def execute_strategy(self):
        # Demonstrate complex decision making
        strategy_crew = Crew(
            agents=[
                Agent(role="Strategy Expert",
                      goal="Develop optimal market strategy")
            ],
            tasks=[
                Task(description="Create detailed strategy based on analysis",
                     expected_output="Step-by-step action plan")
            ]
        )
        return strategy_crew.kickoff()

    @listen(or_("medium_confidence", "low_confidence"))
    def request_additional_analysis(self):
        self.state.recommendations.append("Gather more data")
        return "Additional analysis required"
</code></pre>

<p>This example demonstrates how to:</p>
<ol>
    <li>Use Python code for basic data operations</li>
    <li>Create and execute Crews as steps in your workflow</li>
    <li>Use Flow decorators to manage the sequence of operations</li>
    <li>Implement conditional branching based on Crew results</li>
</ol>

<h2 id="connecting-your-crew-to-a-model">Connecting Your Crew to a Model</h2>
<p>CrewAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.</p>
<p>Please refer to the <a href="https://docs.crewai.com/how-to/LLM-Connections/">Connect CrewAI to LLMs</a> page for details on configuring your agents' connections to models.</p>

<h2 id="how-crewai-compares">How CrewAI Compares</h2>

<p><b>CrewAI's Advantage:</b> CrewAI combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture. The framework excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.</p>
<ul>
    <li><b>LangGraph:</b> While LangGraph provides a foundation for building agent workflows, its approach requires significant boilerplate code and complex state management patterns. The framework's tight coupling with LangChain can limit flexibility when implementing custom agent behaviors or integrating with external systems.</li>
</ul>
<p><i>P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example (<a href="https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent">see comparison</a>) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example (<a href="https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb">detailed analysis</a>).</i></p>
<ul>
    <li><b>Autogen:</b> While Autogen excels at creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.</li>
    <li><b>ChatDev:</b> ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.</li>
</ul>

<h2 id="contribution">Contribution</h2>

<p>CrewAI is open-source and we welcome contributions. If you're looking to contribute, please:</p>
<ul>
    <li>Fork the repository.</li>
    <li>Create a new branch for your feature.</li>
    <li>Add your feature or improvement.</li>
    <li>Send a pull request.</li>
    <li>We appreciate your input!</li>
</ul>

<h3>Installing Dependencies</h3>
<pre><code>uv lock
uv sync
</code></pre>

<h3>Virtual Env</h3>
<pre><code>uv venv
</code></pre>

<h3>Pre-commit hooks</h3>
<pre><code>pre-commit install
</code></pre>

<h3>Running Tests</h3>
<pre><code>uv run pytest .
</code></pre>

<h3>Running static type checks</h3>
<pre><code>uvx mypy src
</code></pre>

<h3>Packaging</h3>
<pre><code>uv build
</code></pre>

<h3>Installing Locally</h3>
<pre><code>pip install dist/*.tar.gz
</code></pre>

<h2 id="telemetry">Telemetry</h2>

<p>CrewAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.</p>
<p>It's pivotal to understand that <b>NO data is collected</b> concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned. When the <code>share_crew</code> feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.</p>
<p>Data collected includes:</p>
<ul>
    <li>Version of CrewAI
        <ul>
            <li>So we can understand how many users are using the latest version</li>
        </ul>
    </li>
    <li>Version of Python
        <ul>
            <li>So we can decide on what versions to better support</li>
        </ul>
    </li>
    <li>General OS (e.g. number of CPUs, macOS/Windows/Linux)
        <ul>
            <li>So we know what OS we should focus on and if we could build specific OS related features</li>
        </ul>
    </li>
    <li>Number of agents and tasks in a crew
        <ul>
            <li>So we make sure we are testing internally with similar use cases and educate people on the best practices</li>
        </ul>
    </li>
    <li>Crew Process being used
        <ul>
            <li>Understand where we should focus our efforts</li>
        </ul>
    </li>
    <li>If Agents are using memory or allowing delegation
        <ul>
            <li>Understand if we improved the features or maybe even drop them</li>
        </ul>
    </li>
    <li>If Tasks are being executed in parallel or sequentially
        <ul>
            <li>Understand if we should focus more on parallel execution</li>
        </ul>
    </li>
    <li>Language model being used
        <ul>
            <li>Improved support on most used languages</li>
        </ul>
    </li>
    <li>Roles of agents in a crew
        <ul>
            <li>Understand high level use cases so we can build better tools, integrations and examples about it</li>
        </ul>
    </li>
    <li>Tools names available
        <ul>
            <li>Understand out of the publicly available tools, which ones are being used the most so we can improve them</li>
        </ul>
    </li>
</ul>
<p>Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the <code>share_crew</code> attribute to <code>True</code> on their Crews. Enabling <code>share_crew</code> results in the collection of detailed crew and task execution data, including <code>goal</code>, <code>backstory</code>, <code>context</code>, and <code>output</code> of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.</p>

<h2 id="license">License</h2>

<p>CrewAI is released under the <a href="https://github.com/crewAIInc/crewAI/blob/main/LICENSE">MIT License</a>.</p>

<h2 id="frequently-asked-questions-faq">Frequently Asked Questions (FAQ)</h2>

<h3>General</h3>
<ul>
  <li><a href="#q-what-exactly-is-crewai">What exactly is CrewAI?</a></li>
  <li><a href="#q-how-do-i-install-crewai">How do I install CrewAI?</a></li>
  <li><a href="#q-does-crewai-depend-on-langchain">Does CrewAI depend on LangChain?</a></li>
  <li><a href="#q-is-crewai-open-source">Is CrewAI open-source?</a></li>
  <li><a href="#q-does-crewai-collect-data-from-users">Does CrewAI collect data from users?</a></li>
</ul>

<h3>Features and Capabilities</h3>
<ul>
  <li><a href="#q-can-crewai-handle-complex-use-cases">Can CrewAI handle complex use cases?</a></li>
  <li><a href="#q-can-i-use-crewai-with-local-ai-models">Can I use CrewAI with local AI models?</a></li>
  <li><a href="#q-what-makes-crews-different-from-flows">What makes Crews different from Flows?</a></li>
  <li><a href="#q-how-is-crewai-better-than-langchain">How is CrewAI better than LangChain?</a></li>
  <li><a href="#q-does-crewai-support-fine-tuning-or-training-custom-models">Does CrewAI support fine-tuning or training custom models?</a></li>
</ul>

<h3>Resources and Community</h3>
<ul>
  <li><a href="#q-where-can-i-find-real-world-crewai-examples">Where can I find real-world CrewAI examples?</a></li>
  <li><a href="#q-how-can-i-contribute-to-crewai">How can I contribute to CrewAI?</a></li>
</ul>

<h3>Enterprise Features</h3>
<ul>
  <li><a href="#q-what-additional-features-does-crewai-enterprise-offer">What additional features does CrewAI Enterprise offer?</a></li>
  <li><a href="#q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments">Is CrewAI Enterprise available for cloud and on-premise deployments?</a></li>
  <li><a href="#q-can-i-try-crewai-enterprise-for-free">Can I try CrewAI Enterprise for free?</a></li>
</ul>

<h3 id="q-what-exactly-is-crewai">Q: What exactly is CrewAI?</h3>
<p>A: CrewAI is a standalone, lean, and fast Python framework built specifically for orchestrating autonomous AI agents. Unlike frameworks like LangChain, CrewAI does not rely on external dependencies, making it leaner, faster, and simpler.</p>

<h3 id="q-how-do-i-install-crewai">Q: How do I install CrewAI?</h3>
<p>A: Install CrewAI using pip:</p>
<pre><code>pip install crewai</code></pre>

<h3 id="q-does-crewai-depend-on-langchain">Q: Does CrewAI depend on LangChain?</h3>
<p>A: No. CrewAI is built entirely from the ground up, with no dependencies on LangChain or other agent frameworks. This ensures a lean, fast, and flexible experience.</p>

<h3 id="q-can-crewai-handle-complex-use-cases">Q: Can CrewAI handle complex use cases?</h3>
<p>A: Yes. CrewAI excels at both simple and highly complex real-world scenarios, offering deep customization options at both high and low levels, from internal prompts to sophisticated workflow orchestration.</p>

<h3 id="q-can-i-use-crewai-with-local-ai-models">Q: Can I use CrewAI with local AI models?</h3>
<p>A: Absolutely! CrewAI supports various language models, including local ones. Tools like Ollama and LM Studio allow seamless integration. Check the <a href="https://docs.crewai.com/how-to/LLM-Connections/">LLM Connections documentation</a> for more details.</p>

<h3 id="q-what-makes-crews-different-from-flows">Q: What makes Crews different from Flows?</h3>
<p>A: Crews provide autonomous agent collaboration, ideal for tasks requiring flexible decision-making and dynamic interaction. Flows offer precise, event-driven control, ideal for managing detailed execution paths and secure state management. You can seamlessly combine both for maximum effectiveness.</p>

<h3 id="q-how-is-crewai-better-than-langchain">Q: How is CrewAI better than LangChain?</h3>