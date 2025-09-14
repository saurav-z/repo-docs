<div align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</div>

<h1 align="center">Codegen: Your AI-Powered Software Engineering Assistant</h1>

<div align="center">
  <a href="https://pypi.org/project/codegen/">
    <img src="https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue" alt="PyPI">
  </a>
  <a href="https://docs.codegen.com">
    <img src="https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square" alt="Documentation">
  </a>
  <a href="https://community.codegen.com">
    <img src="https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square" alt="Slack Community">
  </a>
  <a href="https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file">
    <img src="https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray" alt="License">
  </a>
  <a href="https://x.com/codegen">
    <img src="https://img.shields.io/twitter/follow/codegen?style=social" alt="Follow on X">
  </a>
</div>

<br />

**Tired of repetitive coding tasks? Codegen's SDK lets you harness the power of AI to automate software development and boost your productivity.** This Python SDK provides a programmatic interface to the powerful code agents available from [Codegen](https://codegen.com), offering a seamless way to integrate AI-driven coding assistance into your workflow.

## Key Features

*   **AI-Powered Code Generation:** Generate code, implement new features, and solve coding problems with the help of AI.
*   **Easy Integration:** Simple Python SDK for effortless integration into your existing projects.
*   **Flexible Usage:** Interact with your AI engineer through APIs, Slack, Linear, GitHub, or the Codegen website.
*   **Status Tracking:** Monitor the progress of your tasks and access results upon completion.
*   **Enterprise Ready:** Contact us for enterprise solutions and demos.

## Getting Started

First, install the Codegen SDK:

```bash
pip install codegen
# or
uv pip install codegen
```

Then, obtain your API token from [codegen.com/token](https://codegen.com/token) and your organization ID.

Here's a basic example of how to use the SDK:

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
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please refer to our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up your development environment and submit contributions.

## Enterprise Solutions

For more information on enterprise engagements or to request a demo, please [contact us](https://codegen.com/contact).

---

**[Visit the Codegen GitHub Repository](https://github.com/codegen-sh/codegen) for more details and to contribute to the project.**