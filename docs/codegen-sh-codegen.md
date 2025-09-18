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

**Tired of repetitive coding tasks? The Codegen SDK empowers you to build, test, and deploy software faster by harnessing the power of AI code generation.** This Python SDK provides a programmatic interface to the AI-powered code agents offered by [Codegen](https://codegen.com).  [View the original repository here](https://github.com/codegen-sh/codegen).

## Key Features

*   **AI-Powered Code Generation:** Generate code based on natural language prompts.
*   **API Access:**  Programmatically interact with Codegen's AI agents.
*   **Task Management:** Monitor the status of your code generation tasks.
*   **Easy Installation:**  Install using `pip`, `pipx`, or `uv`.
*   **Automatic Updates:**  Stay up-to-date with the built-in self-update functionality within the CLI.
*   **Integration:**  Interact with your AI engineer via API, Slack, Linear, GitHub, or the Codegen website.

## Getting Started

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

Install the Codegen SDK using your preferred package manager:

```bash
pip install codegen
# or
pipx install codegen
# or
uv tool install codegen
```

## Keeping Up to Date

The Codegen CLI includes a convenient self-update system:

```bash
# Update to the latest version
codegen update

# Check for available updates
codegen update --check

# Update to a specific version
codegen update --version 1.2.3
```

The CLI automatically checks for updates daily.

## Resources

*   [Docs](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up the development environment and submit contributions.

## Enterprise

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).