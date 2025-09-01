<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen SDK: Unleash the Power of AI-Powered Software Engineering</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)

</div>

<br />

**Tired of tedious coding tasks? The Codegen SDK empowers developers to automate code generation and software engineering workflows using powerful AI agents.**

This SDK provides a programmatic interface to interact with the AI-powered code agents offered by [Codegen](https://codegen.com), allowing you to integrate AI code generation into your development pipeline.

## Key Features

*   **Automated Code Generation:** Generate code based on natural language prompts.
*   **AI-Powered Agents:** Leverage intelligent agents to handle complex software engineering tasks.
*   **Easy Integration:** Integrate AI code generation seamlessly into your existing workflows.
*   **Status Tracking:** Monitor the progress of your code generation tasks.
*   **Versatile Results:** Receive code, summaries, and links as results.
*   **Flexible Deployment:** Run agents via API or integrate with platforms like Slack, Linear, and GitHub.

## Getting Started

### Installation

Install the Codegen SDK using pip or uv:

```bash
pip install codegen
# or
uv pip install codegen
```

### Usage

1.  **Get your API Token:** Obtain your organization ID and API token from [codegen.com/token](https://codegen.com/token).
2.  **Initialize the Agent:**

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

*   **Documentation:** [docs.codegen.com](https://docs.codegen.com)
*   **Getting Started:** [docs.codegen.com/introduction/getting-started](https://docs.codegen.com/introduction/getting-started)
*   **Community:** [Join our Slack Community](https://community.codegen.com)
*   **Codegen Website:** [codegen.com](https://codegen.com)
*   **API Token:** [codegen.com/token](https://codegen.com/token)
*   **Original Repository:** [https://github.com/codegen-sh/codegen](https://github.com/codegen-sh/codegen)
*   **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Contact Us:** [codegen.com/contact](https://codegen.com/contact)
*   **Request a Demo:** [codegen.com/request-demo](https://codegen.com/request-demo)

## Contributing

We welcome contributions!  Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting pull requests.

## Enterprise

For information on enterprise engagements and custom solutions, please [contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo).