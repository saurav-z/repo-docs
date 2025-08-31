<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: The AI Software Engineer That Automates Your Coding Tasks</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

Codegen empowers developers with an AI-powered software engineer, automating tasks and accelerating development workflows.  **[Explore the original repository on GitHub](https://github.com/codegen-sh/codegen)**

## Key Features of the Codegen SDK

*   **Automated Code Generation:** Generate code based on natural language prompts.
*   **Task Management:**  Manage and track code generation tasks with status updates.
*   **Flexible Integration:** Integrate with your existing development workflow through API access.
*   **Multi-Platform Support:**  Interact with your AI engineer via API, Slack, Linear, GitHub, or the website.
*   **Easy Installation:** Simple installation using pip or uv.

## Getting Started with Codegen

The Codegen SDK provides a programmatic interface to the code agents provided by [Codegen](https://codegen.com). Here's a quick example:

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

### Installation

Install the SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

### Configuration

1.  **Sign Up:** Get started at [codegen.com](https://codegen.com).
2.  **API Token:** Obtain your API token at [codegen.com/token](https://codegen.com/token).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up the development environment and submitting your contributions.

## Enterprise Solutions

For information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).