<div align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</div>

<h1 align="center">Codegen: Your AI-Powered Software Engineering Companion</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Tired of repetitive coding tasks? Codegen is an AI-powered SDK that empowers you to automate software development and boost your productivity.** This README provides an overview of the Codegen SDK and how you can leverage its capabilities.  Find the original repo at [https://github.com/codegen-sh/codegen](https://github.com/codegen-sh/codegen).

## Key Features

*   **AI-Powered Code Generation:** Utilize advanced AI to generate code, implement features, and automate development tasks.
*   **Programmatic Interface:** Access code agents directly through the Codegen SDK for seamless integration into your workflows.
*   **Easy Installation:**  Get started quickly with straightforward installation via pip, pipx, or uv.
*   **Automated Updates:**  Stay up-to-date with the latest features and improvements with built-in self-update functionality.
*   **Flexible Integration:** Interact with your AI engineer through API, Slack, Linear, GitHub, or the Codegen website.

## Getting Started

The Codegen SDK allows you to interact with code agents provided by [Codegen](https://codegen.com).

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

Install the Codegen SDK using your preferred package manager:

```bash
pip install codegen
# or
pipx install codegen
# or
uv tool install codegen
```

## Keeping Up to Date

The Codegen CLI includes a built-in self-update system:

```bash
# Update to the latest version
codegen update

# Check for available updates
codegen update --check

# Update to a specific version
codegen update --version 1.2.3
```

The CLI automatically checks for updates daily and notifies you when a new version is available.

## Resources

*   [Codegen Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Join the Slack Community](https://community.codegen.com)
*   [Codegen Website](https://codegen.com)

## Contributing

We welcome contributions!  Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting contributions.

## Enterprise Solutions

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).