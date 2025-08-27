<div align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo"/>
  </a>
</div>

<h1 align="center">Codegen SDK: Your AI-Powered Software Engineering Assistant</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br/>

**Supercharge your software development workflow with the Codegen SDK, a powerful Python library providing programmatic access to AI-powered code agents.**  This SDK allows you to automate code generation, debugging, and other software engineering tasks with ease.  [Learn more at the Codegen Github Repository](https://github.com/codegen-sh/codegen).

## Key Features of the Codegen SDK:

*   **Automated Code Generation:** Generate code snippets, features, and entire applications based on natural language prompts.
*   **AI-Powered Code Assistance:**  Leverage intelligent agents for debugging, code optimization, and refactoring.
*   **Seamless Integration:** Easily integrate Codegen's AI capabilities into your existing development environment.
*   **Flexible API Access:** Interact with AI engineers via API.
*   **Multi-Platform Support:** Chat with your AI engineer in Slack, Linear, Github, or on our website.
*   **Easy Installation:** Get started quickly with straightforward pip or uv installation.

## Quick Start

Get up and running in minutes:

```bash
pip install codegen
# or
uv pip install codegen
```

Then, initialize the agent and run your first task:

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

Remember to obtain your API token and organization ID from [codegen.com/token](https://codegen.com/token).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions!  Please review our [Contributing Guide](CONTRIBUTING.md) for details on how to set up your development environment and submit pull requests.

## Enterprise Solutions

For information on enterprise engagements and custom solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).