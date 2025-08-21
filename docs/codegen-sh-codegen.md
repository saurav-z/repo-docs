<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: Revolutionize Software Development with AI</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen empowers you to build software faster by providing a programmatic interface to cutting-edge AI code agents.** ([See the original repository](https://github.com/codegen-sh/codegen))

## Key Features

*   **AI-Powered Code Generation:** Leverage AI to automate coding tasks, from simple features to complex implementations.
*   **API Integration:** Seamlessly integrate Codegen agents into your existing workflows through a user-friendly Python SDK.
*   **Real-Time Status Tracking:** Monitor the progress of your AI agent tasks.
*   **Multi-Platform Support:** Interact with your AI engineer via API, Slack, Linear, GitHub, and our website.
*   **Easy Installation:** Install the SDK with `pip` or `uv`.
*   **Flexible Integration:** Designed to be compatible with various development environments and projects.

## Getting Started

Quickly integrate AI-powered code generation into your workflow.

### Installation

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

### Get Your API Token

Obtain your API token and get started at [codegen.com/token](https://codegen.com/token).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please review our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on setting up your development environment and submitting changes.

## Enterprise Solutions

For inquiries regarding enterprise engagements and custom solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).