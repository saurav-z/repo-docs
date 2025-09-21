<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: The AI-Powered Software Engineer That Never Sleeps</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Revolutionize your software development workflow with Codegen, an AI-powered SDK that empowers you to automate coding tasks and accelerate your projects.**  This Python SDK provides a seamless programmatic interface to the powerful code agents offered by [Codegen](https://codegen.com), allowing you to integrate AI-driven code generation directly into your applications.

## Key Features

*   **AI-Powered Code Generation:** Leverage the power of AI to automate tasks like code generation, feature implementation, and more.
*   **Easy Integration:**  Simple Python SDK for effortless integration into your existing projects.
*   **Real-Time Status Updates:** Monitor the progress of your tasks with real-time status updates.
*   **Flexible Deployment:**  Easily install and update the Codegen CLI.
*   **Multi-Platform Access:** Interact with your AI engineer via API, Slack, Linear, Github, or the Codegen website.

## Getting Started

### Installation

Install the Codegen SDK using pip:

```bash
pip install codegen
# or
pipx install codegen
# or
uv tool install codegen
```

### Usage Example

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

### Keeping Up to Date

The Codegen CLI includes a built-in self-update system:

```bash
# Update to the latest version
codegen update

# Check for available updates
codegen update --check

# Update to a specific version
codegen update --version 1.2.3
```

The CLI automatically checks for updates daily and notifies you when a new version is available.

## Resources

*   [Codegen Documentation](https://docs.codegen.com)
*   [Getting Started Guide](https://docs.codegen.com/introduction/getting-started)
*   [Codegen Community Slack](https://community.codegen.com)
*   [Codegen Website](https://codegen.com)
*   [Contact Us](https://codegen.com/contact)
*   [Original Repository](https://github.com/codegen-sh/codegen)
*   [Contributing Guide](CONTRIBUTING.md)

## Contributing

We welcome contributions! Please refer to our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up your development environment and submit contributions.

## Enterprise

For information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).
```

**Key improvements and SEO optimizations:**

*   **Clear, Concise Headline:**  Includes the target keyword "Codegen" and emphasizes the AI aspect.
*   **One-Sentence Hook:** Immediately grabs the reader's attention and highlights the core benefit.
*   **Keyword Optimization:**  Uses relevant keywords like "AI-powered," "code generation," "SDK," and "automate coding tasks" throughout.
*   **Bulleted Feature List:**  Provides a quick overview of the key benefits in an easy-to-scan format.
*   **Revised Introduction:**  Focuses on the value proposition for the user.
*   **Clear Calls to Action:**  Encourages users to visit the website and get started.
*   **Improved Formatting:** Uses headings, subheadings, and bold text for better readability.
*   **Includes Alt Text:** Added alt text to the image.
*   **Added Link Back:** Explicitly includes a link back to the original repo.
*   **Expanded Documentation Links:** Added more relevant links to the docs.
*   **Contact Information:** Added contact and demo request links.