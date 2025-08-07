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

# PR-Agent: AI-Powered Pull Request Review and Automation

**PR-Agent, the open-source tool, enhances your pull request workflow by providing AI-driven feedback, suggestions, and automation, making code review more efficient and effective.  Find out more at [PR-Agent's GitHub Repo](https://github.com/qodo-ai/pr-agent).**

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Pro App](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![OpenSource App](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
<a href="https://github.com/Codium-ai/pr-agent/commits/main">
<img alt="GitHub" src="https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge" height="20">
</a>

## Table of Contents

- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Why Use PR-Agent?](#why-use-pr-agent)
- [See PR-Agent in Action](#see-it-in-action)
- [Try It Now](#try-it-now)
- [Qodo Merge 💎](#qodo-merge-)
- [How PR-Agent Works](#how-it-works)
- [Data Privacy](#data-privacy)
- [Contributing](#contributing)
- [Links](#links)

## Key Features

*   **Automated Code Review:** Get instant feedback on your code changes.
*   **AI-Powered Suggestions:** Receive helpful suggestions for improvement.
*   **Multi-Platform Support:** Works with GitHub, GitLab, BitBucket, and Azure DevOps.
*   **Multiple Integration Options:** Use it as a CLI, GitHub Action, or Chrome extension.
*   **Customizable Tools:** Tailor the PR-Agent's functionality to your needs.
*   **PR Compression Strategy:** Designed to effectively handle both short and long PRs.

## Getting Started

PR-Agent offers flexible setup options to integrate seamlessly with your workflow:

### Try It Instantly

Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve`.

### GitHub Action

Automate PR reviews with a simple workflow file using the [GitHub Action setup guide](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action).

#### Other Platforms

*   [GitLab webhook setup](https://qodo-merge-docs.qodo.ai/installation/gitlab/)
*   [BitBucket app installation](https://qodo-merge-docs.qodo.ai/installation/bitbucket/)
*   [Azure DevOps setup](https://qodo-merge-docs.qodo.ai/installation/azure/)

### CLI Usage

Run PR-Agent locally on your repository via command line: [Local CLI setup guide](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli).

### Qodo Merge as post-commit in your local IDE

See [here](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

### Discover Qodo Merge 💎

A zero-setup hosted solution with advanced features and priority support.

*   **[FREE for Open Source](https://github.com/marketplace/qodo-merge-pro-for-open-source)**: Full features, zero cost for public repos.
*   [Intro and Installation guide](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/)
*   [Plans & Pricing](https://www.qodo.ai/pricing/)

### Qodo Merge as a Post-commit in Your Local IDE

You can receive automatic feedback from Qodo Merge on your local IDE after each [commit](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

## Why Use PR-Agent?

PR-Agent stands out from other tools because it emphasizes practical, real-world usage with a focus on speed and affordability. Key advantages include:

*   **Fast and Affordable:** Each tool leverages a single LLM call for quick (~30 seconds) and cost-effective results.
*   **Modular and Customizable:** JSON prompting enables easy control over the `/review` tool categories through the configuration file.
*   **Wide Compatibility:** Supports GitHub, GitLab, and BitBucket. Offers various usage options (CLI, GitHub Action, and more).
*   **Multiple Models:** Support for a variety of models including GPT, Claude, and Deepseek.

## See PR-Agent in Action

Here's a glimpse of PR-Agent in action:

#### `/describe`

<div align="center">
<p float="center">
<img src="https://www.codium.ai/images/pr_agent/describe_new_short_main.png" width="512">
</p>
</div>
<hr>

#### `/review`

<div align="center">
<p float="center">
<kbd>
<img src="https://www.codium.ai/images/pr_agent/review_new_short_main.png" width="512">
</kbd>
</p>
</div>
<hr>

#### `/improve`

<div align="center">
<p float="center">
<kbd>
<img src="https://www.codium.ai/images/pr_agent/improve_new_short_main.png" width="512">
</kbd>
</p>
</div>

<hr>

## Try It Now

To start using PR-Agent, simply mention `@CodiumAI-Agent` followed by your desired command in any PR comment on your public GitHub repository.

For example:

```
@CodiumAI-Agent /review
```

The agent will then provide a review of your PR.

**Note:** This is a promotional bot and has limitations. It does not have edit access and cannot be used on private repositories.

## Qodo Merge 💎

[Qodo Merge](https://www.qodo.ai/pricing/) is the hosted, fully managed version of PR-Agent, providing:

1.  **Fully Managed:** We handle hosting, models, updates, and more.
2.  **Enhanced Privacy:** Zero data storage or training on your data. Qodo Merge uses an OpenAI account with zero data retention.
3.  **Priority Support:** Qodo Merge users receive priority support and can request new features.
4.  **Advanced Features:** Emphasizes customization and static code analysis.
    See [here](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) for a list of Qodo Merge's features.

## How PR-Agent Works

The following diagram illustrates PR-Agent tools and their flow:

![PR-Agent Tools](https://www.qodo.ai/images/pr_agent/diagram-v0.9.png)

Learn more about the [PR Compression strategy](https://qodo-merge-docs.qodo.ai/core-abilities/#pr-compression-strategy) used to convert code diffs into manageable LLM prompts.

## Data Privacy

### Self-hosted PR-Agent

-   If you host PR-Agent with your OpenAI API key, it is between you and OpenAI. You can read their API data privacy policy here: [OpenAI Privacy Policy](https://openai.com/enterprise-privacy)

### Qodo-hosted Qodo Merge 💎

-   Qodo Merge 💎 does not store or use your data for training. It also utilizes an OpenAI account with zero data retention.

-   Qodo-hosted Qodo Merge may utilize Qodo's proprietary models for certain clients, with appropriate notification.

-   Qodo Merge is activated only when you invoke it, analyzing only the data relevant to the executed command and pull request.

### Qodo Merge Chrome extension

-   The [Qodo Merge Chrome extension](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl) modifies the visual appearance of the GitHub PR screen without transmitting any user repo or pull request code. Code is sent only when you submit a GitHub comment that activates a PR-Agent tool, in accordance with Qodo-Merge's standard privacy policy.

## Contributing

To contribute to the project, please consult our [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md).

## Links

*   **Discord Community:** [Discord Invite](https://discord.com/invite/SgSxuQ65GF)
*   **Qodo Site:** [Qodo Website](https://www.qodo.ai/)
*   **Blog:** [Qodo Blog](https://www.qodo.ai/blog/)
*   **Troubleshooting:** [Troubleshooting](https://www.qodo.ai/blog/technical-faq-and-troubleshooting/)
*   **Support:** [support@qodo.ai](mailto:support@qodo.ai)