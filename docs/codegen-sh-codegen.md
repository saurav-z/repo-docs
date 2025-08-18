<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">
  Codegen: The AI-Powered Software Engineer for Automated Code Generation
</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen empowers developers to automate coding tasks and accelerate software development with its AI-driven code generation capabilities.** This SDK provides a powerful programmatic interface to interact with Codegen's AI code agents.

## Key Features of the Codegen SDK:

*   **Automated Code Generation:** Leverage AI to generate code based on your prompts and specifications.
*   **API-Driven Interaction:**  Integrate Codegen directly into your workflows via a simple and easy-to-use API.
*   **Flexible Deployment:**  Utilize the SDK and AI agents via API, Slack, Linear, GitHub, or the Codegen website.
*   **Status Tracking:** Monitor the progress of your code generation tasks with real-time status updates.
*   **Easy Integration:**  Simple installation via pip or uv.
*   **Multiple Output Types:** Receive code, summaries, links, and more as results.

## Quickstart: Get Started with Codegen

Here's how to start using the Codegen SDK:

```python
from codegen.agents.agent import Agent

# Initialize the Agent with your organization ID and API token
agent = Agent(
    org_id="YOUR_ORG_ID",  # Find this at codegen.com/developer
    token="YOUR_API_TOKEN",  # Get this from codegen.com/developer
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

## Get Your API Token

To start using Codegen, create an account and get your API token at [codegen.com/developer](https://codegen.com/developer).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Website](https://codegen.com)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on setting up your development environment and submitting pull requests.

## Enterprise Solutions

For enterprise inquiries or to request a demo, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

---

**[View the Original Repository on GitHub](https://github.com/codegen-sh/codegen)**