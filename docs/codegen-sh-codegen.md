<div align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</div>

<h1 align="center">Codegen SDK: Unleash AI-Powered Software Engineering</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)
</div>

<br />

**The Codegen SDK empowers developers with AI-driven code generation and software engineering capabilities, dramatically accelerating your development workflow.** This Python SDK provides a seamless programmatic interface to the powerful AI agents offered by [Codegen](https://codegen.com).

## Key Features

*   **AI-Powered Code Generation:** Generate code snippets, implement new features, and automate coding tasks using AI agents.
*   **API Integration:** Interact with your AI engineer directly through an intuitive API.
*   **Flexible Deployment:** Supports flexible integration of different base URLs.
*   **Real-Time Status Updates:** Track the progress of your tasks and receive status updates.
*   **Multi-Platform Support:** Easily interact with your AI engineer via API, Slack, Linear, Github, or our website.

## Installation

Get started quickly by installing the Codegen SDK:

```bash
pip install codegen
# or
uv pip install codegen
```

## Quick Start

Here's how to begin using the Codegen SDK to request AI-powered code:

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

**Important:** Obtain your organization ID and API token from [codegen.com/token](https://codegen.com/token) to use the SDK.

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Website](https://codegen.com)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please review our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting pull requests.

## Enterprise Solutions

For information about enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

## About Codegen

The Codegen SDK provides a powerful interface to interact with the AI-powered code generation tools offered by Codegen. Visit our [original repository](https://github.com/codegen-sh/codegen) to learn more and contribute to the project!