<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
  <source media="(prefers-color-scheme: light)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
  <img src="https://codium.ai/images/pr_agent/logo-light.png" alt="logo" width="330">
</picture>
<br/>

[Installation Guide](https://qodo-merge-docs.qodo.ai/installation/) |
[Usage Guide](https://qodo-merge-docs.qodo.ai/usage-guide/) |
[Tools Guide](https://qodo-merge-docs.qodo.ai/tools/) |
[Qodo Merge](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) 💎

</div>

# PR-Agent: AI-Powered Pull Request Automation

**PR-Agent streamlines your code review process by providing AI-driven feedback and suggestions, helping you improve code quality and accelerate your workflow.** [Visit the original repository](https://github.com/qodo-ai/pr-agent) for more details.

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Pro App](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![Open Source App](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
<a href="https://github.com/Codium-ai/pr-agent/commits/main">
<img alt="GitHub" src="https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge" height="20">
</a>

## Table of Contents

*   [Key Features](#key-features)
*   [Getting Started](#getting-started)
    *   [Try It Instantly](#try-it-instantly)
    *   [GitHub Action](#github-action)
    *   [Other Platforms](#other-platforms)
    *   [CLI Usage](#cli-usage)
    *   [Qodo Merge in Your IDE](#qodo-merge-in-your-ide)
    *   [Discover Qodo Merge 💎](#discover-qodo-merge-)
*   [News and Updates](#news-and-updates)
*   [Why Use PR-Agent?](#why-use-pr-agent)
*   [See It in Action](#see-it-in-action)
*   [Qodo Merge 💎](#qodo-merge-)
*   [How It Works](#how-it-works)
*   [Data Privacy](#data-privacy)
*   [Contributing](#contributing)
*   [Links](#links)

## Key Features

*   **Automated Code Reviews:** Get instant feedback and suggestions on your pull requests.
*   **Multiple Git Provider Support:** Works seamlessly with GitHub, GitLab, Bitbucket, and Azure DevOps.
*   **Flexible Integration:** Use PR-Agent via CLI, GitHub Actions, GitHub App, or Docker.
*   **Customizable Tools:** Modular and customizable tools with configurable categories.
*   **PR Compression Strategy:** Effectively handles both short and long pull requests.
*   **Qodo Merge Integration:** Utilize advanced features through the hosted Qodo Merge platform.

## Getting Started

### Try It Instantly

Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve`

### GitHub Action

Automate PR reviews using a simple workflow file.  Refer to the [GitHub Action setup guide](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action).

### Other Platforms

*   [GitLab webhook setup](https://qodo-merge-docs.qodo.ai/installation/gitlab/)
*   [BitBucket app installation](https://qodo-merge-docs.qodo.ai/installation/bitbucket/)
*   [Azure DevOps setup](https://qodo-merge-docs.qodo.ai/installation/azure/)

### CLI Usage

Run PR-Agent locally via command line. See [Local CLI setup guide](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli).

### Qodo Merge in Your IDE

Receive automatic feedback from Qodo Merge on your local IDE after each [commit](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

### Discover Qodo Merge 💎

Zero-setup hosted solution with advanced features and priority support
*   **[FREE for Open Source](https://github.com/marketplace/qodo-merge-pro-for-open-source)**: Full features, zero cost for public repos
*   [Intro and Installation guide](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/)
*   [Plans & Pricing](https://www.qodo.ai/pricing/)

## News and Updates

*(Summarized for brevity - original content in the original README)*

*   **Jul 1, 2025:** Automatic feedback in local IDE after commit.
*   **Jun 21, 2025:** v0.30 released.
*   **Jun 3, 2025:** Simplified free tier for Qodo Merge.
*   **Apr 30, 2025:** Chat on code suggestions feature for Qodo Merge's `/improve` tool.
*   **Apr 16, 2025:** New `/scan_repo_discussions` tool for Qodo Merge.

## Why Use PR-Agent?

PR-Agent is designed for practical team usage, offering:

*   **Fast Results:** Each tool utilizes a single LLM call for quick responses.
*   **Effective PR Handling:**  The core PR Compression strategy handles various PR lengths.
*   **Customization:** Modular tools with customizable categories via configuration.
*   **Wide Compatibility:**  Supports multiple Git providers and usage methods.

## See It in Action

*(Screenshots are included in the original README and should be included here as well)*

*   `/describe` -  *Insert /describe screenshot here*
*   `/review` -  *Insert /review screenshot here*
*   `/improve` -  *Insert /improve screenshot here*

## Qodo Merge 💎

[Qodo Merge](https://www.qodo.ai/pricing/) is a hosted, paid version of PR-Agent by Qodo, offering:

1.  **Fully Managed:**  Qodo handles hosting, models, and updates.
2.  **Improved Privacy:** No data storage or model training with your data.
3.  **Improved Support:** Priority support and feature requests.
4.  **Extra Features:**  Increased customization and static code analysis.
    See [Qodo Merge features](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/)

## How It Works

*(Diagram of PR-Agent tools and flow should be included here)*

See the [PR Compression strategy](https://qodo-merge-docs.qodo.ai/core-abilities/#pr-compression-strategy) page for details on how code diffs are converted into prompts.

## Data Privacy

### Self-hosted PR-Agent

*   Your data privacy is governed by OpenAI's API data privacy policy:  [https://openai.com/enterprise-privacy](https://openai.com/enterprise-privacy)

### Qodo-hosted Qodo Merge 💎

*   Qodo Merge does not store, or train on your data.
*   For certain clients, Qodo-hosted Qodo Merge will use Qodo’s proprietary models — if this is the case, you will be notified.
*   No passive data collection; PR-Agent is only active when you invoke it.

### Qodo Merge Chrome extension

*   The Chrome extension modifies the GitHub PR screen's visual appearance. It does not transmit any user's repo or pull request code, but is governed by the standard privacy policy of Qodo-Merge.

## Contributing

Contribute to the project by reading our [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md).

## Links

*   Discord community: https://discord.com/invite/SgSxuQ65GF
*   Qodo site: https://www.qodo.ai/
*   Blog: https://www.qodo.ai/blog/
*   Troubleshooting: https://www.qodo.ai/blog/technical-faq-and-troubleshooting/
*   Support: support@qodo.ai