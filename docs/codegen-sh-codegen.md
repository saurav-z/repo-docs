<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen SDK: Automate Your Software Development with AI</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**The Codegen SDK empowers developers to harness the power of AI, automating code generation and streamlining software development.** This SDK provides a programmatic interface to interact with Codegen's powerful code agents, letting you automate tasks from feature implementation to code reviews.

## Key Features

*   **AI-Powered Code Generation:** Leverage intelligent agents to generate code based on your prompts.
*   **Flexible Integration:** Integrate seamlessly into your existing workflows via a Python SDK.
*   **Asynchronous Task Management:** Monitor and manage long-running tasks with status updates.
*   **Easy Installation:** Simple installation via pip or uv.
*   **Multi-Platform Support:** Interact with your AI engineer via API, Slack, Linear, GitHub, or our website.
*   **Robust Documentation:** Access comprehensive documentation to get you started quickly.

## Getting Started

Install the Codegen SDK using pip:

```bash
pip install codegen
```

or with uv:

```bash
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

Get your API token and organization ID at [codegen.com/token](https://codegen.com/token) and learn more at [codegen.com](https://codegen.com).

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Contributing](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions!  Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting pull requests.

## Enterprise

For information on enterprise solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

---

**[View the original repository on GitHub](https://github.com/codegen-sh/codegen)**
```

Key improvements and SEO considerations:

*   **Strong Title & Hook:**  Uses a clear title and a concise one-sentence hook that immediately conveys the value proposition ("Automate Your Software Development with AI").
*   **Keywords:** Includes relevant keywords like "AI," "code generation," "SDK," and "automation" to improve search visibility.
*   **Clear Headings:** Uses descriptive headings and subheadings to structure the content, making it easier to read and understand.
*   **Bulleted Key Features:**  Highlights the main benefits of the SDK using bullet points, allowing for quick scanning.
*   **Concise Language:**  Uses clear and concise language throughout the description.
*   **Call to Actions:** Encourages users to get started and provides clear links to resources.
*   **Alt Text:** Added alt text to the image to improve SEO
*   **GitHub Link:** Added a link back to the original repo at the end, as requested.
*   **Organization:** Reorganized and streamlined the content for better readability and flow.