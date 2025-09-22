<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: The AI Software Engineer That Automates Code Generation</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen empowers developers to automate software development tasks with the power of AI, streamlining your workflow and boosting productivity.** This SDK provides a Python interface to interact with Codegen's AI code agents, allowing you to automate code generation, feature implementation, and more.

## Key Features

*   **Automated Code Generation:** Generate code based on natural language prompts.
*   **Feature Implementation:**  Implement new features and functionalities with ease.
*   **Status Tracking:** Monitor the progress of your AI-powered tasks.
*   **Flexible Integration:** Interact with Codegen through API, Slack, Linear, GitHub, and more.
*   **Self-Updating CLI:** Stay up-to-date with the latest features and improvements via the built-in update system.

## Getting Started

The Codegen SDK is easy to integrate into your projects.

### Installation

Install the SDK using `pip`, `pipx`, or `uv`:

```bash
pip install codegen
# or
pipx install codegen
# or
uv tool install codegen
```

### Usage

Here's a quick example to get you started:

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

## Keeping Up to Date

Update the Codegen CLI to the latest version using the built-in update system:

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
*   [API Token](https://codegen.com/token)
*   [Contributing](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

See our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up the development environment and submitting contributions.

## Enterprise

For information on enterprise engagements, [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

---

**[View the original repository on GitHub](https://github.com/codegen-sh/codegen)**