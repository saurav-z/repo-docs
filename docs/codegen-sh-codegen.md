<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen SDK: Empower Your Development with AI</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen's SDK provides a powerful programmatic interface to leverage AI-powered code generation, transforming how you build and innovate.**  

## Key Features

*   **AI-Powered Code Generation:** Generate code, implement new features, and automate development tasks using the Codegen AI engine.
*   **Programmatic Access:** Interact with the Codegen AI through a simple and intuitive Python SDK.
*   **Seamless Integration:** Easily integrate Codegen into your existing workflows.
*   **Flexible Deployment:** Utilize the SDK with your preferred tools and environments.
*   **Real-time Status Updates:** Monitor the progress of your tasks with real-time status updates.
*   **Multi-Platform Support:** Interact with your AI engineer via API, Slack, Linear, Github, or the Codegen website.

## Getting Started

### Installation

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

### Usage Example

Here's how to get started with the Codegen SDK:

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

*   **Documentation:** [https://docs.codegen.com](https://docs.codegen.com)
*   **Getting Started:** [https://docs.codegen.com/introduction/getting-started](https://docs.codegen.com/introduction/getting-started)
*   **Original Repository:** [https://github.com/codegen-sh/codegen](https://github.com/codegen-sh/codegen)
*   **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Contact Us:** [https://codegen.com/contact](https://codegen.com/contact)

## Contributing

Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up the development environment and submit contributions.

## Enterprise

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).
```

Key improvements and explanations:

*   **SEO-Friendly Title and Description:**  Uses the keywords "Codegen SDK," "AI," and "code generation" to attract users searching for these terms.  The initial hook sentence clearly states the value proposition.
*   **Clear Headings:** Organizes the content with clear headings (Key Features, Getting Started, Resources, etc.) for readability and searchability.
*   **Bulleted Key Features:**  Highlights the main selling points of the SDK in an easily digestible bulleted list.
*   **Concise Language:** Streamlines the text for better clarity and engagement.
*   **Call to Action:** Encourages users to get started with direct links.
*   **Link to Original Repo:** Added a link to the original GitHub repository for easy access to the source code and related information.
*   **Alt text:**  Added "alt" text to the image for accessibility.
*   **Removed Redundancy:** Removed some of the redundant introductory text.