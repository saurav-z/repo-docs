<div align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</div>

<h1 align="center">Codegen SDK: Your AI-Powered Software Engineering Assistant</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Tired of repetitive coding tasks?** The Codegen SDK provides a powerful programmatic interface to AI-powered code agents, helping you automate software development and accelerate your workflow.  Visit the [original repo](https://github.com/codegen-sh/codegen) for more information.

## Key Features

*   **AI-Powered Code Generation:** Leverage AI to automate code creation, modification, and optimization.
*   **Programmatic Interface:** Interact with Codegen's AI agents directly through a Python SDK.
*   **Simple Integration:** Easily integrate Codegen into your existing development workflows.
*   **Status Tracking:** Monitor the progress of your AI-driven tasks with clear status updates.
*   **Multi-Platform Support:** Interact with your AI engineer via API, Slack, Linear, Github, or on our website.

## Getting Started

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

### Example Usage

Here's a quick example of how to get started:

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

Obtain your API token and organization ID by visiting [codegen.com/token](https://codegen.com/token).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please refer to our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting your contributions.

## Enterprise Solutions

For inquiries regarding enterprise engagements and custom solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).