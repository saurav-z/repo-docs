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

**Codegen's SDK provides a powerful programmatic interface to AI-powered code agents, enabling you to automate software development tasks with ease.**  This README provides a guide to the Codegen SDK; learn more and see the code for the SDK on its original [GitHub repository](https://github.com/codegen-sh/codegen).

## Key Features of Codegen

*   **AI-Powered Code Generation:** Leverage advanced AI to generate code, implement features, and automate software development workflows.
*   **Easy Integration:** Seamlessly integrate the Codegen SDK into your existing projects.
*   **Flexible API:** Interact with code agents through a simple and intuitive API.
*   **Real-time Status Updates:** Monitor the progress of your tasks with detailed status updates.
*   **Multi-Platform Support:** Engage with your AI engineer via API, Slack, Linear, GitHub, and our website.
*   **Automate Software Engineering tasks**: Sort Users by Last Login with AI

## Getting Started with the Codegen SDK

Install the SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

Here's how to get started:

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

1.  **Sign Up:** Get started at [codegen.com](https://codegen.com).
2.  **Get your API Token:** Obtain your API token at [codegen.com/token](https://codegen.com/token).
3.  **Start Coding:** Use the provided code example to integrate the Codegen SDK into your project.

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Contributing](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up the development environment and submitting contributions.

## Enterprise Solutions

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).