<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen SDK: Supercharge Your Development with AI-Powered Code Generation</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Tired of repetitive coding tasks? The Codegen SDK empowers you with AI-driven code generation, letting you focus on innovation.** This Python SDK provides a seamless interface to interact with the powerful code agents offered by [Codegen](https://codegen.com), helping you automate development workflows and boost productivity.

## Key Features

*   **AI-Powered Code Generation:** Leverage cutting-edge AI to generate code snippets, entire features, and more, based on simple prompts.
*   **Easy Integration:** Integrate the SDK into your existing projects with just a few lines of Python code.
*   **Multi-Platform Support:** Interact with your AI engineer via API, or chat with it in Slack, Linear, Github, or on our website.
*   **Status Tracking:** Monitor the progress of your code generation tasks and easily access results.
*   **Flexible & Customizable:** Tailor the agents to your specific needs and development environment.
*   **Secure:** Utilizes secure API keys for authentication.

## Getting Started

Quickly integrate AI-powered code generation into your workflow.

### Installation

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

### Usage Example

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

## API Keys and Setup

1.  **Get an API Key:** Visit [codegen.com/token](https://codegen.com/token) to obtain your API token and organization ID.
2.  **Initialize the Agent:** Use your organization ID and API token to initialize the `Agent` class.
3.  **Run Code Generation Tasks:** Provide prompts to the `agent.run()` method to generate code.

## Resources

*   [Codegen Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Website](https://codegen.com)
*   [Contribute to Codegen](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting contributions.

## Enterprise Solutions

For information on enterprise engagements or to request a demo, please [contact us](https://codegen.com/contact).

**[View the original repository on GitHub](https://github.com/codegen-sh/codegen)**