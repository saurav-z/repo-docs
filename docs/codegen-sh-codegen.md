<div align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</div>

<h1 align="center">Codegen: The AI-Powered Software Engineer</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen empowers developers with an AI-driven SDK to automate software development, streamline workflows, and boost productivity.**  This SDK provides a programmatic interface to the AI code agents provided by [Codegen](https://codegen.com).

## Key Features of the Codegen SDK

*   **AI-Powered Code Generation:**  Leverage advanced AI to generate code based on natural language prompts.
*   **Automated Tasks:** Automate software engineering tasks, from feature implementation to code analysis.
*   **Flexible Integration:**  Integrate Codegen agents into your existing workflows through a simple API.
*   **Real-time Status Updates:**  Monitor the progress of tasks and receive updates on completion.
*   **Multi-Platform Support:** Interact with your AI engineer through API, Slack, Linear, GitHub, and the Codegen website.

## Getting Started with the Codegen SDK

Install the SDK using pip:

```bash
pip install codegen
```

Or using `uv`:

```bash
uv pip install codegen
```

Then, get your API token and organization ID at [codegen.com/token](https://codegen.com/token).

Here's a basic example of how to use the SDK:

```python
from codegen.agents.agent import Agent

# Initialize the Agent with your organization ID and API token
agent = Agent(
    org_id="YOUR_ORG_ID",  # Find this at codegen.com/token
    token="YOUR_API_TOKEN",  # Get this from codegen.com/token
    # base_url="https://codegen-sh-rest-api.modal.run",  # Optional - defaults to production
)

# Run an agent with a prompt
task = agent.run(prompt="Implement a new feature to sort users by last login.")

# Check the initial status
print(task.status)

# Refresh the task to get updated status (tasks can take time)
task.refresh()

# Check the updated status
print(task.status)

# Once task is complete, you can access the result
if task.status == "completed":
    print(task.result)  # Result often contains code, summaries, or links
```

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up your development environment and submit contributions.

## Enterprise Solutions

For enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

## Learn More

For more information on Codegen and its AI-powered software engineering capabilities, visit our website at [codegen.com](https://codegen.com). You can also check out the original repository: [https://github.com/codegen-sh/codegen](https://github.com/codegen-sh/codegen).