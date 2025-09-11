<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: The AI-Powered Software Engineer That Revolutionizes Development</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

Codegen empowers developers to automate software development tasks and accelerate their workflows, offering a powerful SDK and versatile integration options.  **[See the original repo here](https://github.com/codegen-sh/codegen).**

## Key Features of Codegen

*   **AI-Powered Code Generation:** Leverage the power of AI to generate code based on your prompts and specifications.
*   **Programmatic Interface:** Utilize the Codegen SDK for seamless integration into your existing workflows and applications.
*   **Multi-Platform Support:** Interact with your AI engineer via API, Slack, Linear, Github, or directly on the website.
*   **Simplified Development:** Automate tasks, reduce development time, and increase overall productivity.
*   **Easy Installation:** Get started quickly with a simple `pip install codegen` command.
*   **Comprehensive Documentation:**  Access detailed documentation to guide you through the SDK and its capabilities.

## Getting Started with Codegen

The Codegen SDK allows you to interact with AI-powered code agents. Here's a quick example:

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

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

### Configuration

1.  **Get Started:** Sign up at [codegen.com](https://codegen.com).
2.  **Get Your API Token:** Obtain your API token at [codegen.com/token](https://codegen.com/token).
3.  **Use the SDK:**  Use the code example above, replacing `"YOUR_ORG_ID"` and `"YOUR_API_TOKEN"` with your actual credentials.

## Resources

*   [Documentation](https://docs.codegen.com)
*   [Getting Started](https://docs.codegen.com/introduction/getting-started)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up the development environment and submitting contributions.

## Enterprise Solutions

For more information on enterprise engagements or to request a demo, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).
```
Key improvements and SEO optimizations:

*   **Headline Optimization:**  The title is more descriptive and includes relevant keywords ("AI-Powered," "Software Engineer," "Development").
*   **One-Sentence Hook:**  Immediately explains the core benefit of Codegen.
*   **Bulleted Key Features:** Makes the core benefits immediately visible.
*   **Clearer Call to Action:** "Getting Started" section.
*   **SEO Keywords:** Used terms like "AI-Powered," "code generation," "SDK," and "software development" throughout the text.
*   **Internal Linking:** Added links to the documentation and other resources.
*   **Concise Formatting:** Improved the readability with clear headings and short paragraphs.
*   **Alt Text:** added to image.
*   **Original Repo Link:** Added.