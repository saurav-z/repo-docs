<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
  <source media="(prefers-color-scheme: light)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
  <img src="https://codium.ai/images/pr_agent/logo-light.png" alt="PR-Agent Logo" width="330">
</picture>
<br/>

[Installation Guide](https://qodo-merge-docs.qodo.ai/installation/) |
[Usage Guide](https://qodo-merge-docs.qodo.ai/usage-guide/) |
[Tools Guide](https://qodo-merge-docs.qodo.ai/tools/) |
[Qodo Merge](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) 💎

</div>

## PR-Agent: AI-Powered Pull Request Automation for Smarter Code Reviews

PR-Agent streamlines your pull request workflow with AI-powered tools for efficient code review, suggestions, and more. ([View the original repo](https://github.com/qodo-ai/pr-agent))

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Pro App](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![Open Source App](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
<a href="https://github.com/Codium-ai/pr-agent/commits/main">
<img alt="GitHub" src="https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge" height="20">
</a>

## Key Features

*   **Automated Code Reviews:** Get AI-powered reviews that highlight potential issues and suggest improvements.
*   **Comprehensive Toolset:** Utilize tools for describing, reviewing, improving, and more.
*   **Multi-Platform Support:** Works seamlessly with GitHub, GitLab, BitBucket, Azure DevOps, and Gitea.
*   **Multiple Integration Options:** Integrate through CLI, GitHub Actions, Webhooks, and IDE extensions.
*   **Customizable and Modular:** Configure and extend tools to fit your specific needs.

## Getting Started

### Quick Start:
Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve`

### Installation and Usage

*   **GitHub Action:** Automate PR reviews with a simple workflow file. [GitHub Action Setup](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action)
*   **GitLab, BitBucket, Azure DevOps, Gitea:**  Set up PR-Agent for your preferred platform [Installation Guides](https://qodo-merge-docs.qodo.ai/installation/)
*   **CLI:** Run PR-Agent locally for immediate feedback.  [Local CLI Setup](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli)

### Leverage Qodo Merge 💎 for Advanced Features

[Qodo Merge](https://www.qodo.ai/pricing/) offers a hosted, feature-rich version with zero-setup and priority support.

*   **Free for Open Source:** Full features, zero cost for public repos. [Qodo Merge for Open Source](https://github.com/marketplace/qodo-merge-pro-for-open-source)
*   **Post-Commit IDE Integration:** Receive automatic feedback in your IDE. [IDE Integration](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)
*   **Learn More:** [Plans and Pricing](https://www.qodo.ai/pricing/)

## Tools Overview

PR-Agent offers a robust suite of tools to enhance your pull request workflow.

| Functionality                     | Description                                                                                                                                  |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `/describe`                       | Automatically generates a pull request description.                                                                                            |
| `/review`                         | Provides a detailed code review with suggestions.                                                                                              |
| `/improve`                        | Offers code improvements and refactoring suggestions.                                                                                          |
| `/ask`                            | Allows you to ask questions about the code.                                                                                                    |
| `/help_docs`                      | Generates or updates documentation.                                                                                                            |
| `/update_changelog`               | Automatically updates the changelog.                                                                                                            |
| `/add_documentation` 💎          | Adds comprehensive documentation to your pull request.                                                                                           |
| `/analyze` 💎                     | Analyzes the impact of the pull request.                                                                                                      |
| `/auto_approve` 💎                | Automatically approves pull requests based on criteria.                                                                                        |
| `/ci_feedback` 💎                 | Provides feedback based on CI/CD results.                                                                                                       |
| `/compliance` 💎                   | Checks the code for compliance with specific standards.                                                                                       |
| `/custom_prompt` 💎                | Allows you to use custom prompts to tailor results.                                                                                          |
| `/generate_custom_labels` 💎        | Generates custom labels based on the pull request content.                                                                                       |
| `/generate_tests` 💎                | Generates tests for the code changes.                                                                                                        |
| `/implement` 💎                   | Implements code changes based on suggestions.                                                                                                |
| `/scan_repo_discussions` 💎         | Scans repository discussions for relevant information.                                                                                         |
| `/similar_code` 💎                 | Finds similar code within the repository.                                                                                                      |
| `/pr_to_ticket` 💎                | Links the pull request to a ticket.                                                                                                         |
| *💎* =  Qodo Merge Exclusive                                     |                                                                    |

## News and Updates

*   **Aug 8, 2025:** Full GPT-5 support added.  See benchmark results [here](https://qodo-merge-docs.qodo.ai/pr_benchmark/#pr-benchmark-results).
*   **Jul 1, 2025:** Post-commit feedback available in your local IDE. [Details](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit).
*   **Jun 21, 2025:** v0.30 [released](https://github.com/qodo-ai/pr-agent/releases).
*   **Jun 3, 2025:** Qodo Merge offers a simplified free tier.
*   **Apr 30, 2025:** `/improve` tool in Qodo Merge now has chat-on-code suggestions.
*   **Apr 16, 2025:** New tool for Qodo Merge 💎 - `/scan_repo_discussions`.

## Why Use PR-Agent?

PR-Agent stands out due to its focus on practicality and efficiency, specifically:

*   **Fast and Affordable:** Designed for quick (approx. 30 seconds) and cost-effective feedback.
*   **PR Compression:** Effectively handles both short and long pull requests with a core compression strategy.
*   **Modular and Customizable:** JSON prompting enables modular tools and easy configuration.
*   **Multi-Platform and Model Support:** Works with GitHub, GitLab, BitBucket, various models, and multiple deployment options.

## See It in Action

*   `/describe` [Example](https://github.com/Codium-ai/pr-agent/pull/530)
*   `/review` [Example](https://github.com/Codium-ai/pr-agent/pull/732#issuecomment-1975099151)
*   `/improve` [Example](https://github.com/Codium-ai/pr-agent/pull/732#issuecomment-1975099159)

## Data Privacy

### Self-hosted PR-Agent
When self-hosting and using your own OpenAI API key, OpenAI's data privacy policy applies.

### Qodo-hosted Qodo Merge 💎

*   No data storage or training.
*   Zero data retention OpenAI account.
*   Qodo Merge uses Qodo’s proprietary models for some clients, and you will be notified.
*   Active only when invoked, analyzing only relevant data.

### Qodo Merge Chrome extension

*   The Chrome extension only modifies the visual appearance of a GitHub PR screen and does not transmit any user's repo or pull request code.

## Contributing

Contribute by following our [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md).

## Links

*   **Discord:** [Discord community](https://discord.com/invite/SgSxuQ65GF)
*   **Qodo Site:** [Qodo](https://www.qodo.ai/)
*   **Blog:** [Qodo Blog](https://www.qodo.ai/blog/)
*   **Troubleshooting:** [Troubleshooting](https://www.qodo.ai/blog/technical-faq-and-troubleshooting/)
*   **Support:** support@qodo.ai