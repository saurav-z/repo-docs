<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: The AI-Powered Software Engineering Assistant</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen empowers developers with an AI-driven SDK, automating code generation and software engineering tasks, so you can ship faster.** This Python SDK provides a programmatic interface to the powerful code agents offered by [Codegen](https://codegen.com).

## Key Features of the Codegen SDK

*   **Automated Code Generation:** Generate code based on natural language prompts, significantly accelerating development.
*   **API-Driven Interaction:** Interact with your AI engineer directly through an intuitive API.
*   **Task Management:** Monitor the status of your code generation tasks and access results seamlessly.
*   **Integration with Popular Platforms:** Integrate Codegen with your workflow through Slack, Linear, GitHub, and more.
*   **Flexible Deployment:** Configure the SDK with your organization ID, API token, and optional base URL for deployment flexibility.
*   **Comprehensive Documentation:** Access detailed documentation to guide you through setup and usage.

## Quick Start: Get Started with Codegen

Here's how to get started with the Codegen SDK:

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

## Installation

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

## Resources

*   [Official Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Website](https://codegen.com)
*   [Contact Us](https://codegen.com/contact)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Original Repository](https://github.com/codegen-sh/codegen)

## Contributing

We welcome contributions! Please review our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting contributions.

## Enterprise Solutions

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).