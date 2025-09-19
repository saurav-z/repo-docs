<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: Your AI-Powered Software Engineer</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Tired of repetitive coding tasks?  The Codegen SDK gives you programmatic access to AI-powered code generation agents, automating your software development workflow.**

## Key Features of the Codegen SDK:

*   **AI-Powered Code Generation:** Leverage AI agents to handle coding tasks.
*   **Simple API Integration:** Easily integrate with your existing development tools and workflows using the Codegen SDK.
*   **Task Management:** Monitor the status of your code generation requests.
*   **Automated Updates:**  Keep your SDK up-to-date with built-in self-update functionality.
*   **Multi-Platform Support:** Available via pip, pipx, and uv.
*   **Integrations:** Interact with Codegen via API, Slack, Linear, Github, and the website.

## Getting Started with Codegen

Here's a quick example of how to use the Codegen SDK:

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

## Staying Up-to-Date

Keep your Codegen CLI updated easily:

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

*   [Codegen Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Website](https://codegen.com)
*   [Codegen API Token](https://codegen.com/token)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contribute

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to set up your development environment and submit contributions.

## Enterprise Solutions

For information about enterprise solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

---
**[View the original repository on GitHub](https://github.com/codegen-sh/codegen)**