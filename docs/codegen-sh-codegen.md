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

**Codegen's SDK empowers developers to harness the power of AI to automate and accelerate the software development lifecycle.**  This allows you to quickly implement features, debug code, and more.  Visit the [Codegen GitHub repository](https://github.com/codegen-sh/codegen) for the latest updates and to contribute.

## Key Features

*   **AI-Powered Code Generation:** Automate code creation, modification, and debugging tasks.
*   **Programmatic Interface:** Interact with Codegen's AI agents directly through an easy-to-use Python SDK.
*   **Flexible Integration:** Integrate with existing development workflows via API access.
*   **Real-time Status Updates:** Monitor the progress of your tasks.
*   **Multiple Platform Support:** Interact with your AI engineer via API, Slack, Linear, Github, and on the website.

## Getting Started

Install the Codegen SDK and start automating your development tasks.

```bash
pip install codegen
# or
uv pip install codegen
```

Then, obtain your API token and organization ID from [codegen.com/token](https://codegen.com/token).

**Example Usage:**

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

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Website](https://codegen.com)
*   [API Token](https://codegen.com/token)
*   [Contributing](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)
*   [Request a Demo](https://codegen.com/request-demo)

## Contributing

Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up the development environment and submit contributions.

## Enterprise

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).