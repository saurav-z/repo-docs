<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo"/>
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

**Codegen empowers you to automate software development tasks with the power of AI, acting as your always-on, intelligent software engineer.** This Python SDK provides a programmatic interface to the Codegen platform, allowing you to integrate AI-driven code generation and automation directly into your workflows.

## Key Features

*   **AI-Powered Code Generation:** Leverage advanced AI models to generate code, implement new features, and more, based on natural language prompts.
*   **API Integration:** Seamlessly integrate Codegen into your existing development tools and workflows using the Python SDK.
*   **Status Tracking:** Monitor the progress of your code generation tasks with clear status updates.
*   **Versatile Output:** Receive results that include code, summaries, links, and more, tailored to your specifications.
*   **Multi-Platform Interaction:** Interact with your AI engineer through various channels, including API, Slack, Linear, GitHub, and the Codegen website.

## Quick Start

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

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

## Getting Started

1.  **Sign up:** Get started at [codegen.com](https://codegen.com).
2.  **Get your API token:** Obtain your API token from [codegen.com/token](https://codegen.com/token).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting pull requests.

## Enterprise Solutions

For information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

## Learn More

Discover the full capabilities of Codegen and how it can transform your development workflow.  Explore the original repository on [GitHub](https://github.com/codegen-sh/codegen).