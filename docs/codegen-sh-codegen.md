<p align="center">
  <a href="https://docs.codegen.com">
    <img src="https://i.imgur.com/6RF9W0z.jpeg" alt="Codegen Logo" />
  </a>
</p>

<h1 align="center">Codegen: Your AI-Powered Software Engineer</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/PyPi-codegen-gray?style=flat-square&color=blue)](https://pypi.org/project/codegen/)
[![Documentation](https://img.shields.io/badge/Docs-docs.codegen.com-purple?style=flat-square)](https://docs.codegen.com)
[![Slack Community](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=flat-square)](https://community.codegen.com)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-gray?&color=gray)](https://github.com/codegen-sh/codegen-sdk/tree/develop?tab=Apache-2.0-1-ov-file)
[![Follow on X](https://img.shields.io/twitter/follow/codegen?style=social)](https://x.com/codegen)
[Link to Original Repo](https://github.com/codegen-sh/codegen)

</div>

<br />

**Codegen empowers developers to automate software engineering tasks with its powerful AI agents, helping you build, test, and deploy code faster than ever before.**

## Key Features

*   **AI-Powered Code Generation:** Leverage AI to generate code based on natural language prompts.
*   **Automated Tasks:** Automate tasks like code implementation, testing, and debugging.
*   **API Integration:** Integrate Codegen's AI agents directly into your development workflow through a programmatic interface.
*   **Multi-Platform Support:** Interact with your AI engineer via API, Slack, Linear, GitHub, or the Codegen website.
*   **Easy Installation:** Simple installation via pip or uv.
*   **Enterprise Solutions:** Tailored solutions for large-scale software development projects.

## Getting Started

1.  **Installation:**

    ```bash
    pip install codegen
    # or
    uv pip install codegen
    ```

2.  **Get your API credentials:** Sign up at [codegen.com](https://codegen.com) and obtain your API token from [codegen.com/token](https://codegen.com/token).

3.  **Example Usage:**

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

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on how to set up your development environment and submit contributions.

## Enterprise Solutions

Looking for a customized solution for your enterprise?  [Contact us](https://codegen.com/contact) or [request a demo](https://codegen.com/request-demo) to learn more.
```
Key improvements and SEO considerations:

*   **Clear Headline:** Replaced the marketing headline with a descriptive and SEO-friendly one, "Codegen: Your AI-Powered Software Engineer."
*   **One-Sentence Hook:** Added a concise, engaging introductory sentence to immediately inform the user about the product's value.
*   **Keyword Optimization:**  Incorporated relevant keywords like "AI," "code generation," "software engineer," "automation," and "API" to improve search visibility.
*   **Bulleted Key Features:** Presented the key features in a concise, easily scannable bulleted list. This helps users quickly understand the value proposition.
*   **Structured Format:**  Organized the content with clear headings and subheadings for improved readability and SEO.
*   **Call to Action:** The "Getting Started" section includes clear instructions on how to install the SDK and get started.
*   **Resource Links:** Reorganized and improved the resource links.
*   **Emphasis on Benefits:** The key features are framed to highlight the benefits (e.g., "Automated Tasks" instead of "Task Automation").
*   **Link back to original repo**: Added a link back to the original repo.
*   **Alt Text for Image:** Added `alt="Codegen Logo"` to the image tag for better accessibility and SEO.