<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen SDK: Automate Software Development with AI</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Tired of repetitive coding tasks?** The Codegen SDK provides a powerful Python interface to interact with AI-powered code agents, enabling you to automate software development workflows.

## Key Features:

*   **AI-Powered Code Generation:** Leverage advanced AI to generate code based on natural language prompts.
*   **Easy Integration:**  Simple Python SDK for seamless integration into your existing projects.
*   **Flexible Usage:** Interact with your AI engineer via API, Slack, Linear, Github, or on our website.
*   **Real-time Status Updates:** Monitor the progress of your tasks with status updates and refresh capabilities.
*   **Multi-platform support:** Use Codegen in your favorite development environments, including Slack, Github, and Linear.

## Getting Started

Install the Codegen SDK using pip:

```bash
pip install codegen
```

To start using the SDK, you'll need an API token and organization ID, which you can obtain from the [Codegen Developer Portal](https://codegen.com/developer).

Here's a quick example:

```python
from codegen.agents.agent import Agent

# Initialize the Agent with your organization ID and API token
agent = Agent(
    org_id="YOUR_ORG_ID",  # Find this at codegen.com/developer
    token="YOUR_API_TOKEN",  # Get this from codegen.com/developer
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
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Website](https://codegen.com)
*   [Codegen GitHub Repository](https://github.com/codegen-sh/codegen)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions!  Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting pull requests.

## Enterprise

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).