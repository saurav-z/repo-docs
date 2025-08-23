<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">
  Codegen SDK: Supercharge Your Development with AI-Powered Code Agents
</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)
</div>

<br />

**The Codegen SDK empowers developers to automate coding tasks and accelerate software development using powerful, AI-driven code agents.**

This Python SDK provides a programmatic interface to interact with the AI code agents offered by [Codegen](https://codegen.com), enabling you to automate code generation, bug fixing, and more.

## Key Features

*   **AI-Powered Code Agents:** Leverage intelligent agents to handle various coding tasks.
*   **Easy Integration:** Simple Python API for seamless integration into your existing workflows.
*   **Task Management:** Monitor the status of your tasks and access results when ready.
*   **Versatile Use Cases:** Automate tasks like new feature implementation, bug fixing, and more.
*   **Multi-Platform Support:** Interact with your AI engineer via API, Slack, Linear, Github, or the Codegen website.

## Getting Started

### Installation

Install the Codegen SDK using pip:

```bash
pip install codegen
```

or using uv:

```bash
uv pip install codegen
```

### Usage Example

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

## Resources

*   **[Documentation](https://docs.codegen.com):** Comprehensive documentation for the Codegen SDK.
*   **[Getting Started](https://docs.codegen.com/introduction/getting-started):** Quickstart guide to help you begin.
*   **[Codegen Website](https://codegen.com):** Learn more about Codegen and its AI-powered solutions.
*   **[API Token](https://codegen.com/token):** Get your API token.
*   **[Slack Community](https://community.codegen.com):** Join the community and engage with other users.
*   **[Contributing Guide](CONTRIBUTING.md):** Contribute to the project.
*   **[Contact Us](https://codegen.com/contact):** Contact the Codegen team.
*   **[Request a Demo](https://codegen.com/request-demo):** Request a demo for enterprise solutions.

## Contributing

We welcome contributions! Please refer to our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting contributions.

## Enterprise Solutions

For information on enterprise engagements and customized solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

---

**[View the original repository on GitHub](https://github.com/codegen-sh/codegen)**