<div align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</div>

<h1 align="center">Codegen: Your AI Software Engineer</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

**Codegen empowers you to automate your software development workflow by leveraging the power of AI.**  This SDK provides a programmatic interface for interacting with the Codegen platform, allowing you to easily integrate AI-powered code generation, feature implementation, and more into your projects.  Visit the [original repo](https://github.com/codegen-sh/codegen) for more details.

## Key Features

*   **AI-Powered Code Generation:** Automate the creation of code from natural language prompts.
*   **Feature Implementation:**  Delegate tasks such as implementing new features directly to the AI agent.
*   **Seamless Integration:**  Integrate the Codegen AI engineer with your existing projects via API.
*   **Flexible Deployment:** Use agents via API or via integrations with Slack, Linear, GitHub, and the Codegen website.
*   **Easy to Use SDK:** A Python SDK is provided for simple integration and control.
*   **Status Tracking:**  Monitor the progress of your tasks and retrieve results upon completion.

## Getting Started

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

Next, get your API token and organization ID from the Codegen website.  Then, you can start using the SDK:

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
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up the development environment and submit contributions.

## Enterprise

For information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).