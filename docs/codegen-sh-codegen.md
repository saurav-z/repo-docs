<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen SDK: Unleash the Power of AI for Software Development</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**The Codegen SDK empowers developers with an AI-powered coding assistant that helps you write, debug, and improve code more efficiently.** This SDK provides a programmatic interface to the powerful code agents available through [Codegen](https://codegen.com), offering seamless integration into your existing development workflow.

## Key Features

*   **AI-Powered Code Generation:** Leverage AI to generate code snippets, implement new features, and automate repetitive tasks.
*   **Easy Integration:** Integrate Codegen's AI agents directly into your Python applications with a simple and intuitive SDK.
*   **Task Management:** Track the status and progress of your code generation tasks.
*   **Flexible Deployment:** Supports various installation methods, including pip, pipx, and uv.
*   **Automated Updates:** Stay up-to-date with the latest features and improvements using the built-in self-update system.

## Getting Started

Install the Codegen SDK using your preferred package manager:

```bash
pip install codegen
# or
pipx install codegen
# or
uv tool install codegen
```

### Usage Example

Here's a quick example of how to use the SDK:

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

Get your API token and organization ID at [codegen.com/token](https://codegen.com/token) and explore the full potential of Codegen at [codegen.com](https://codegen.com).

## Staying Up-to-Date

Keep your SDK up-to-date with the Codegen CLI:

```bash
# Update to the latest version
codegen update

# Check for available updates
codegen update --check

# Update to a specific version
codegen update --version 1.2.3
```

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions!  Please refer to our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up your development environment and submit pull requests.

## Enterprise

For information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

**[Back to Original Repository](https://github.com/codegen-sh/codegen)**