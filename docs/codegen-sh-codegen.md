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

**Codegen empowers developers to automate coding tasks with AI, boosting productivity and accelerating software development.** This SDK provides a programmatic interface to the powerful code generation agents offered by [Codegen](https://codegen.com).

## Key Features

*   **AI-Powered Code Generation:** Leverage AI to automate tasks like feature implementation, code summarization, and more.
*   **Easy Integration:**  Seamlessly integrate Codegen into your existing workflows with a straightforward Python SDK.
*   **Real-time Status Updates:** Monitor task progress and receive updates on the status of your requests.
*   **Flexible Access:** Interact with your AI engineer via API, Slack, Linear, Github, or directly on the Codegen website.
*   **Supports various platforms:** Use with existing projects or start new ones in supported environments.

## Quick Start: Getting Started with Codegen

Here's how to quickly get started using the Codegen SDK:

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

Install the Codegen SDK using `pip` or `uv`:

```bash
pip install codegen
# or
uv pip install codegen
```

## Getting Your API Token

1.  Go to [codegen.com](https://codegen.com) to get started.
2.  Obtain your API token from [codegen.com/token](https://codegen.com/token).
3.  Use your API token to start your project.

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Contributing](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please refer to our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting changes.

## Enterprise Solutions

For information on enterprise engagements and custom solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

## Additional Information

*   [Original Repository](https://github.com/codegen-sh/codegen)