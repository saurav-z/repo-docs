<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: The AI-Powered Code Generation SDK</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Tired of repetitive coding tasks?** The Codegen SDK provides a powerful programmatic interface to AI-powered code agents, enabling you to automate code generation, streamline development, and accelerate your software projects. Explore the [original repository](https://github.com/codegen-sh/codegen) for more details.

## Key Features of the Codegen SDK

*   **AI-Driven Code Generation:** Leverage advanced AI models to generate code from natural language prompts.
*   **Easy Integration:** Seamlessly integrate with your existing workflows using a Python SDK.
*   **Task Management:** Monitor the status of your code generation tasks with detailed progress updates.
*   **Versatile Output:** Receive code, summaries, and links tailored to your requests.
*   **Multiple Integration Points:** Interact with your AI engineer via API, Slack, Linear, Github, and our website.

## Getting Started

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

### Example Usage:

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

To begin, sign up at [codegen.com](https://codegen.com) and retrieve your API token from [codegen.com/token](https://codegen.com/token).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up your development environment and submit your contributions.

## Enterprise Solutions

For information on enterprise engagements and custom solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).