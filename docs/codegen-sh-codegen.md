<div align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</div>

<h1 align="center">Codegen: Your AI-Powered Software Engineering Assistant</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen empowers developers to automate coding tasks, accelerate development cycles, and boost productivity with the help of AI.**  This SDK provides a powerful Python interface to the Codegen platform, enabling you to harness the capabilities of AI-powered code agents directly within your workflows.  You can also interact with your AI engineer via API, or chat with it in Slack, Linear, Github, or on our website.  For more details, see the original repo: [https://github.com/codegen-sh/codegen](https://github.com/codegen-sh/codegen)

## Key Features

*   **AI-Powered Code Generation:** Leverage AI agents to generate code based on natural language prompts.
*   **Automated Task Execution:**  Run agents to implement new features, fix bugs, and perform other software engineering tasks.
*   **Status Tracking:** Monitor the progress of your tasks with clear status updates.
*   **Flexible Integration:**  Integrate Codegen into your existing development environment.
*   **API Access:** Access the Codegen AI engineer via API for programmatic control.
*   **Multiple Interaction Methods:** Chat with the AI in Slack, Linear, GitHub, or on the Codegen website.
*   **Result Retrieval:** Access code, summaries, and links once tasks are complete.

## Getting Started

Quickly integrate AI into your development process with the Codegen SDK.

**Installation:**

```bash
pip install codegen
# or
uv pip install codegen
```

**Usage Example:**

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

**Obtain API Credentials:**

*   Get started at [codegen.com](https://codegen.com) and get your API token at [codegen.com/token](https://codegen.com/token).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions!  Please review our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting pull requests.

## Enterprise

For information on enterprise solutions, [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).