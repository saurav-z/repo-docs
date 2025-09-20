<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: Your AI-Powered Software Engineering Assistant</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen empowers you to automate software development tasks with the power of AI, streamlining your workflow and accelerating your projects.** This Python SDK provides a programmatic interface to the intelligent code agents offered by [Codegen](https://codegen.com).

## Key Features of the Codegen SDK

*   **AI-Powered Code Generation:** Leverage AI to generate code, implement new features, and more.
*   **Simple Integration:** Easy to install and integrate into your existing development environment.
*   **Asynchronous Task Management:** Monitor the progress of your AI-powered tasks with status updates.
*   **Flexible Deployment:** Supports pip, pipx, and uv installation methods.
*   **Automatic Updates:** Stay up-to-date with the latest features and improvements with the built-in self-update system.
*   **Supports Multiple Platforms:** Interact with your AI engineer via API, Slack, Linear, Github, or our website.

## Getting Started

Install the Codegen SDK using your preferred package manager:

```bash
pip install codegen
# or
pipx install codegen
# or
uv tool install codegen
```

Then, initialize the `Agent` with your organization ID and API token (available at [codegen.com/token](https://codegen.com/token)):

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

The Codegen CLI includes a built-in self-update system to ensure you're always running the latest version:

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
*   [Codegen Slack Community](https://community.codegen.com)
*   [Codegen Website](https://codegen.com)
*   [Contact Us](https://codegen.com/contact)
*   [Original Repository](https://github.com/codegen-sh/codegen)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up the development environment and submit contributions.

## Enterprise Solutions

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).