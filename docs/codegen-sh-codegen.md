<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: Unleash the Power of AI-Powered Code Generation</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Codegen empowers you to automate your software development workflow with cutting-edge AI, acting as your tireless software engineering companion.** The Codegen SDK provides a simple yet powerful programmatic interface to interact with Codegen's AI-powered code generation agents, allowing you to integrate intelligent code assistance directly into your projects.  [See the original repository](https://github.com/codegen-sh/codegen).

## Key Features

*   **AI-Powered Code Generation:** Leverage advanced AI to generate code based on your specifications.
*   **Seamless Integration:** Easily integrate Codegen's AI agents into your existing development workflow.
*   **Multi-Platform Support:** Interact with your AI engineer via API, Slack, Linear, GitHub, or the Codegen website.
*   **Simple API:** A clean and intuitive Python API makes it easy to start generating code.
*   **Status Tracking:** Monitor the progress of your code generation tasks.
*   **Enterprise Ready:**  Available for enterprise deployments, contact us for more information.

## Getting Started

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

Then, get your API token and organization ID at [codegen.com/token](https://codegen.com/token).

Here's a quick example of how to use the SDK:

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

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up the development environment and submit your contributions.

## Enterprise

For more information on enterprise engagements, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).
```
Key improvements and SEO considerations:

*   **Clear Hook:**  The opening sentence is designed to grab attention and clearly state the core benefit.
*   **SEO-Friendly Title:**  Includes keywords like "Code Generation" and "AI-Powered" to improve search visibility.
*   **Bulleted Key Features:** Makes it easy to quickly understand the main advantages of using Codegen.
*   **Concise and Readable:**  The text is more streamlined and easier to scan.
*   **Stronger Calls to Action:** Encourages users to explore the documentation and contribute.
*   **Alt Text for Image:** Added `alt` text to the image tag for accessibility.
*   **Structured Headings:** Uses `h1` and `h2` tags for better organization and SEO.
*   **Link to the original repo**: Added a link back to the original repository.