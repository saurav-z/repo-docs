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

**Codegen empowers developers to automate software development tasks, boosting productivity and accelerating project timelines.** This SDK provides a programmatic interface to the powerful code agents offered by [Codegen](https://codegen.com), your AI-powered software engineering assistant.

## Key Features of the Codegen SDK

*   **AI-Powered Code Generation:** Leverage advanced AI to generate code snippets, implement features, and more.
*   **API Access:** Interact with your AI engineer directly through a simple and intuitive API.
*   **Task Management:** Monitor the progress of your tasks and retrieve results once complete.
*   **Flexible Integration:** Integrate with your existing workflows and development tools.
*   **Multi-Platform Support:** Access your AI engineer through various platforms, including Slack, Linear, GitHub, and the Codegen website.

## Getting Started

Here's a quick example of how to get started with the Codegen SDK:

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

Install the SDK using pip:

```bash
pip install codegen
```

or

```bash
uv pip install codegen
```

### Configuration

1.  **Sign up and get an API token:** Get started at [codegen.com](https://codegen.com) and obtain your API token at [codegen.com/token](https://codegen.com/token).

## Resources

*   **Documentation:** [https://docs.codegen.com](https://docs.codegen.com)
*   **Getting Started:** [https://docs.codegen.com/introduction/getting-started](https://docs.codegen.com/introduction/getting-started)
*   **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Contact Us:** [https://codegen.com/contact](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up your development environment and submit pull requests.

## Enterprise Solutions

For information on enterprise engagements and custom solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

**Original Repo:**  [https://github.com/codegen-sh/codegen](https://github.com/codegen-sh/codegen)