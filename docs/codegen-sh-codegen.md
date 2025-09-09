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

**Tired of tedious coding tasks? The Codegen SDK empowers you to leverage AI-powered software engineering, automating development and boosting productivity.**

## Key Features

*   **AI-Powered Code Generation:** Generate code from prompts, specifications, and more.
*   **Automated Feature Implementation:**  Delegate feature development to AI agents.
*   **Seamless Integration:**  Easily integrate with your existing workflows via a Python SDK.
*   **Status Monitoring:**  Track the progress of your tasks and receive updates.
*   **Versatile Output:**  Get code, summaries, and other valuable outputs from AI agents.
*   **Flexible Interaction:** Interact with your AI engineer through API, Slack, Linear, Github or via our website.

## Getting Started

The Codegen SDK provides a programmatic interface to the code agents offered by [Codegen](https://codegen.com).  Quickly automate your software development tasks and see your productivity soar.

### Installation

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

### Usage Example

Here's how to get started using the Codegen SDK:

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

*   [Codegen Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Website](https://codegen.com)
*   [Get Your API Token](https://codegen.com/token)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions!  Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting changes.

## Enterprise Solutions

Looking for advanced features and support? For information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

---

**[Visit the original repository on GitHub](https://github.com/codegen-sh/codegen)**
```
Key improvements and SEO considerations:

*   **Strong Headline:**  Uses a compelling headline with primary keywords ("Codegen SDK," "Automate," "Software Development," "AI").
*   **One-Sentence Hook:** Grabs attention and clearly states the value proposition.
*   **Key Feature Section:** Uses bullet points for easy readability and highlights core functionalities.
*   **Clear "Getting Started" Section:** Provides installation and a practical usage example.
*   **Keyword Optimization:** Uses relevant keywords throughout the README (e.g., "AI," "code generation," "automation").
*   **Calls to Action:** Encourages users to explore resources and contribute.
*   **Organization and Formatting:**  Uses headings, subheadings, and spacing for improved readability and clarity.
*   **Included Alt Text:** Image alt text is included.
*   **Link Back to Original Repo:** Adds a clear link back to the original repository at the end for users to find the source code.