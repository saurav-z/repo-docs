<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">
  Codegen: Your AI-Powered Software Engineering Assistant
</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Tired of repetitive coding tasks? Codegen's AI-powered SDK streamlines your software development workflow, helping you build faster and smarter.**  This Python SDK provides a programmatic interface to the powerful code agents offered by [Codegen](https://codegen.com), empowering developers to automate coding tasks, generate code, and accelerate their projects.

## Key Features

*   **AI-Powered Code Generation:** Leverage advanced AI to automate code creation based on prompts.
*   **Easy Integration:** Simple Python SDK for seamless integration into your existing projects.
*   **Task Management:** Monitor and manage code generation tasks with status updates and result retrieval.
*   **Flexible Deployment:** Interact with your AI engineer through APIs, Slack, Linear, Github, or our website.
*   **Collaboration:** Collaborate with your AI-powered assistant across multiple platforms.

## Getting Started

1.  **Installation:** Install the SDK using `pip` or `uv`:

    ```bash
    pip install codegen
    # or
    uv pip install codegen
    ```

2.  **Authentication:**  Get your API token and organization ID from [codegen.com/token](https://codegen.com/token).

3.  **Usage Example:**

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
*   [Codegen Website](https://codegen.com)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Contact Us](https://codegen.com/contact)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up your development environment and submit changes.

## Enterprise Solutions

For enterprise inquiries and demo requests, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).

## Further Information

For more details, visit the original repository on GitHub: [https://github.com/codegen-sh/codegen](https://github.com/codegen-sh/codegen)