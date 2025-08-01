<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
  <source media="(prefers-color-scheme: light)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
  <img src="https://codium.ai/images/pr_agent/logo-light.png" alt="PR-Agent Logo" width="330">
</picture>
</div>

# PR-Agent: AI-Powered Pull Request Automation 🤖

**PR-Agent automates your pull request workflow with AI-powered features, improving efficiency and code quality.** Check out the [original repo](https://github.com/qodo-ai/pr-agent) for more details!

[Installation Guide](https://qodo-merge-docs.qodo.ai/installation/) |
[Usage Guide](https://qodo-merge-docs.qodo.ai/usage-guide/) |
[Tools Guide](https://qodo-merge-docs.qodo.ai/tools/) |
[Qodo Merge Pro](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) 💎

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Qodo Merge Pro App](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![Open Source App](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
[![Last Commit](https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge&height=20)](https://github.com/Codium-ai/pr-agent/commits/main)

## Table of Contents

- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Why Use PR-Agent?](#why-use-pr-agent)
- [Qodo Merge 💎](#qodo-merge-)
- [How It Works](#how-it-works)
- [Data Privacy](#data-privacy)
- [Contributing](#contributing)
- [Links](#links)

## Key Features

*   **Automated PR Analysis:** Generate comprehensive descriptions, reviews, and improvement suggestions using AI.
*   **Multi-Platform Support:** Works with GitHub, GitLab, Bitbucket, and Azure DevOps.
*   **Flexible Usage:** Integrate via CLI, GitHub Actions, or webhooks.
*   **Customizable Tools:** Tailor the AI's behavior to your project's specific needs.
*   **PR Compression:**  Handles both short and long PRs.
*   **Multiple Model Support:** Supports various LLM models like GPT, Claude, and Deepseek.
*   **Comprehensive Feature Set**: Features like `Describe`, `Review`, `Improve`, `Ask`, and much more.

## Getting Started

### Try It Instantly

Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve` on a pull request.

### GitHub Action

Automate PR reviews by adding a simple workflow file to your repository. See the [GitHub Action setup guide](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action).

#### Other Platform Integrations

*   [GitLab webhook setup](https://qodo-merge-docs.qodo.ai/installation/gitlab/)
*   [BitBucket app installation](https://qodo-merge-docs.qodo.ai/installation/bitbucket/)
*   [Azure DevOps setup](https://qodo-merge-docs.qodo.ai/installation/azure/)

### CLI Usage

Run PR-Agent locally on your repository via command line. See the [Local CLI setup guide](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli).

### Qodo Merge as post-commit in your local IDE

See [here](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

### Discover Qodo Merge 💎

Zero-setup hosted solution with advanced features and priority support
-  **[FREE for Open Source](https://github.com/marketplace/qodo-merge-pro-for-open-source)**: Full features, zero cost for public repos
-  [Intro and Installation guide](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/)
-  [Plans & Pricing](https://www.qodo.ai/pricing/)

### Qodo Merge as a Post-commit in Your Local IDE
You can receive automatic feedback from Qodo Merge on your local IDE after each [commit](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)


## News and Updates

### Jul 1, 2025

You can now receive automatic feedback from Qodo Merge in your local IDE after each commit. Read more about it [here](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit).

### Jun 21, 2025

v0.30 was [released](https://github.com/qodo-ai/pr-agent/releases)

### Jun 3, 2025

Qodo Merge now offers a simplified free tier 💎.
Organizations can use Qodo Merge at no cost, with a [monthly limit](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/#cloud-users) of 75 PR reviews per organization.

### Apr 30, 2025

A new feature is now available in the `/improve` tool for Qodo Merge 💎 - Chat on code suggestions.

<img width="512" alt="image" src="https://codium.ai/images/pr_agent/improve_chat_on_code_suggestions_ask.png" />

Read more about it [here](https://qodo-merge-docs.qodo.ai/tools/improve/#chat-on-code-suggestions).

### Apr 16, 2025

New tool for Qodo Merge 💎 - `/scan_repo_discussions`.

<img width="635" alt="image" src="https://codium.ai/images/pr_agent/scan_repo_discussions_2.png" />

Read more about it [here](https://qodo-merge-docs.qodo.ai/tools/scan_repo_discussions/).

## Why Use PR-Agent?

PR-Agent streamlines your pull request process, offering fast, affordable, and practical AI-powered assistance. Here's why it stands out:

*   **Focus on Practicality:**  Each tool (review, improve, etc.) uses a single LLM call for quick and cost-effective results.
*   **PR Compression Strategy:** Efficiently handles both short and long PRs.
*   **Modular & Customizable:**  JSON prompting allows for easy customization of tools like the '/review' tool categories via a configuration file.
*   **Cross-Platform:** Supports GitHub, GitLab, BitBucket, and multiple usage methods (CLI, GitHub Action, GitHub App, Docker, etc.).
*   **Multiple Model Support:** Supports various LLM models.

## Features

PR-Agent and Qodo Merge provide a wide array of features, seamlessly integrated with various Git providers.

|                                                         |                                                                                        | GitHub | GitLab | Bitbucket | Azure DevOps | Gitea |
|---------------------------------------------------------|----------------------------------------------------------------------------------------|:------:|:------:|:---------:|:------------:|:-----:|
| [TOOLS](https://qodo-merge-docs.qodo.ai/tools/)         | [Describe](https://qodo-merge-docs.qodo.ai/tools/describe/)                            |   ✅   |   ✅   |    ✅     |      ✅      |  ✅   |
|                                                         | [Review](https://qodo-merge-docs.qodo.ai/tools/review/)                                |   ✅   |   ✅   |    ✅     |      ✅      |  ✅   |
|                                                         | [Improve](https://qodo-merge-docs.qodo.ai/tools/improve/)                              |   ✅   |   ✅   |    ✅     |      ✅      |  ✅   |
|                                                         | [Ask](https://qodo-merge-docs.qodo.ai/tools/ask/)                                      |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | ⮑ [Ask on code lines](https://qodo-merge-docs.qodo.ai/tools/ask/#ask-lines)            |   ✅   |   ✅   |           |              |       |
|                                                         | [Help Docs](https://qodo-merge-docs.qodo.ai/tools/help_docs/?h=auto#auto-approval)     |   ✅   |   ✅   |    ✅     |              |       |
|                                                         | [Update CHANGELOG](https://qodo-merge-docs.qodo.ai/tools/update_changelog/)            |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | [Add Documentation](https://qodo-merge-docs.qodo.ai/tools/documentation/) 💎           |   ✅   |   ✅   |           |              |       |
|                                                         | [Analyze](https://qodo-merge-docs.qodo.ai/tools/analyze/) 💎                           |   ✅   |   ✅   |           |              |       |
|                                                         | [Auto-Approve](https://qodo-merge-docs.qodo.ai/tools/improve/?h=auto#auto-approval) 💎 |   ✅   |   ✅   |    ✅     |              |       |
|                                                         | [CI Feedback](https://qodo-merge-docs.qodo.ai/tools/ci_feedback/) 💎                   |   ✅   |        |           |              |       |
|                                                         | [Compliance](https://qodo-merge-docs.qodo.ai/tools/compliance/) 💎                     |   ✅   |    ✅   |    ✅     |              |       |
|                                                         | [Custom Prompt](https://qodo-merge-docs.qodo.ai/tools/custom_prompt/) 💎                                            |   ✅   |   ✅   |    ✅     |              |       |
|                                                         | [Generate Custom Labels](https://qodo-merge-docs.qodo.ai/tools/custom_labels/) 💎                                   |   ✅   |   ✅   |           |              |       |
|                                                         | [Generate Tests](https://qodo-merge-docs.qodo.ai/tools/test/) 💎                                                    |   ✅   |   ✅   |           |              |       |
|                                                         | [Implement](https://qodo-merge-docs.qodo.ai/tools/implement/) 💎                                                    |   ✅   |   ✅   |    ✅     |              |       |
|                                                         | [Scan Repo Discussions](https://qodo-merge-docs.qodo.ai/tools/scan_repo_discussions/) 💎                            |   ✅   |        |           |              |       |
|                                                         | [Similar Code](https://qodo-merge-docs.qodo.ai/tools/similar_code/) 💎                                              |   ✅   |        |           |              |       |
|                                                         | [Utilizing Best Practices](https://qodo-merge-docs.qodo.ai/tools/improve/#best-practices) 💎                        |   ✅   |   ✅   |    ✅     |              |       |
|                                                         | [PR Chat](https://qodo-merge-docs.qodo.ai/chrome-extension/features/#pr-chat) 💎                                    |   ✅   |        |           |              |       |
|                                                         | [PR to Ticket](https://qodo-merge-docs.qodo.ai/tools/pr_to_ticket/) 💎                                              |   ✅   |   ✅   |    ✅     |              |       |
|                                                         | [Suggestion Tracking](https://qodo-merge-docs.qodo.ai/tools/improve/#suggestion-tracking) 💎                        |   ✅   |   ✅   |           |              |       |
|                                                         |                                                                                                                     |        |        |           |              |       |
| [USAGE](https://qodo-merge-docs.qodo.ai/usage-guide/)   | [CLI](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli)                            |   ✅   |   ✅   |    ✅     |      ✅      |  ✅   |
|                                                         | [App / webhook](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#github-app)                      |   ✅   |   ✅   |    ✅     |      ✅      |  ✅   |
|                                                         | [Tagging bot](https://github.com/Codium-ai/pr-agent#try-it-now)                                                     |   ✅   |        |           |              |       |
|                                                         | [Actions](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action)                              |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         |                                                                                                                     |        |        |           |              |       |
| [CORE](https://qodo-merge-docs.qodo.ai/core-abilities/) | [Adaptive and token-aware file patch fitting](https://qodo-merge-docs.qodo.ai/core-abilities/compression_strategy/) |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | [Auto Best Practices 💎](https://qodo-merge-docs.qodo.ai/core-abilities/auto_best_practices/)                       |   ✅   |      |         |            |   |
|                                                         | [Chat on code suggestions](https://qodo-merge-docs.qodo.ai/core-abilities/chat_on_code_suggestions/)                |   ✅   |  ✅   |           |              |       |
|                                                         | [Code Validation 💎](https://qodo-merge-docs.qodo.ai/core-abilities/code_validation/)                               |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | [Dynamic context](https://qodo-merge-docs.qodo.ai/core-abilities/dynamic_context/)                                  |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | [Fetching ticket context](https://qodo-merge-docs.qodo.ai/core-abilities/fetching_ticket_context/)                  |   ✅    |  ✅    |     ✅     |              |       |
|                                                         | [Global and wiki configurations](https://qodo-merge-docs.qodo.ai/usage-guide/configuration_options/) 💎             |   ✅   |   ✅   |    ✅     |              |       |
|                                                         | [Impact Evaluation](https://qodo-merge-docs.qodo.ai/core-abilities/impact_evaluation/) 💎                           |   ✅   |   ✅   |           |              |       |
|                                                         | [Incremental Update](https://qodo-merge-docs.qodo.ai/core-abilities/incremental_update/)                            |   ✅    |       |           |              |       |
|                                                         | [Interactivity](https://qodo-merge-docs.qodo.ai/core-abilities/interactivity/)                                      |   ✅   |  ✅   |           |              |       |
|                                                         | [Local and global metadata](https://qodo-merge-docs.qodo.ai/core-abilities/metadata/)                               |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | [Multiple models support](https://qodo-merge-docs.qodo.ai/usage-guide/changing_a_model/)                            |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | [PR compression](https://qodo-merge-docs.qodo.ai/core-abilities/compression_strategy/)                              |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | [PR interactive actions](https://www.qodo.ai/images/pr_agent/pr-actions.mp4) 💎                                     |   ✅   |   ✅   |           |              |       |
|                                                         | [RAG context enrichment](https://qodo-merge-docs.qodo.ai/core-abilities/rag_context_enrichment/)                    |   ✅    |       |    ✅     |              |       |
|                                                         | [Self reflection](https://qodo-merge-docs.qodo.ai/core-abilities/self_reflection/)                                  |   ✅   |   ✅   |    ✅     |      ✅      |       |
|                                                         | [Static code analysis](https://qodo-merge-docs.qodo.ai/core-abilities/static_code_analysis/) 💎                     |   ✅   |   ✅   |           |              |       |
- 💎 means this feature is available only in [Qodo Merge](https://www.qodo.ai/pricing/)

## See It in Action

**`/describe`**

<div align="center">
<p float="center">
<img src="https://www.codium.ai/images/pr_agent/describe_new_short_main.png" width="512">
</p>
</div>
<hr>

**`/review`**

<div align="center">
<p float="center">
<kbd>
<img src="https://www.codium.ai/images/pr_agent/review_new_short_main.png" width="512">
</kbd>
</p>
</div>
<hr>

**`/improve`**

<div align="center">
<p float="center">
<kbd>
<img src="https://www.codium.ai/images/pr_agent/improve_new_short_main.png" width="512">
</kbd>
</p>
</div>

## Try It Now

To experiment with PR-Agent, mention `@CodiumAI-Agent` in any PR comment and add the desired command. For example:

```
@CodiumAI-Agent /review
```

The agent will then provide a review of your PR.  Note that this is a promotional bot and has limited capabilities.

---

## Qodo Merge 💎

[Qodo Merge](https://www.qodo.ai/pricing/) offers a fully managed, hosted version of PR-Agent with premium features.

Key benefits:

1.  **Fully Managed:**  Qodo handles hosting, model updates, and more. Installation is as simple as adding the Qodo Merge app to your repository.
2.  **Enhanced Privacy:** No data is stored or used to train models. Qodo Merge uses an OpenAI account with zero data retention.
3.  **Priority Support:**  Qodo Merge users receive priority support and can request new features.
4.  **Advanced Features:** Includes increased customization and static code analysis, along with LLM logic, to provide improved results.

See [Qodo Merge features](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) for more details.

## How It Works

PR-Agent utilizes the following flow:

![PR-Agent Tools](https://www.qodo.ai/images/pr_agent/diagram-v0.9.png)

For details on the PR Compression strategy, visit [PR Compression strategy](https://qodo-merge-docs.qodo.ai/core-abilities/#pr-compression-strategy).

## Data Privacy

### Self-hosted PR-Agent

*   If you host PR-Agent with your OpenAI API key, you are subject to OpenAI's privacy policy:  [OpenAI's privacy policy](https://openai.com/enterprise-privacy)

### Qodo-hosted Qodo Merge 💎

*   Qodo Merge does not store or use your data for training purposes. It uses an OpenAI account with zero data retention.
*   Qodo Merge will only be active when invoked.  It extracts and analyzes only data relevant to the executed command and queried pull request.
*   For some clients, Qodo Merge uses Qodo’s proprietary models, and users will be notified if this is the case.

### Qodo Merge Chrome extension

*   The [Qodo Merge Chrome extension](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl) modifies the visual appearance of GitHub PR screens.  It does not transmit any user's repo or pull request code. Code is only sent for processing when a user submits a GitHub comment that activates a PR-Agent tool, in accordance with the standard privacy policy of Qodo-Merge.

## Contributing

Contributions are welcome!  See our [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md) to get started.

## Links

*   Discord: [Discord community](https://discord.com/invite/SgSxuQ65GF)
*   Qodo: [Qodo](https://www.qodo.ai/)
*   Blog: [Qodo Blog](https://www.qodo.ai/blog/)
*   Troubleshooting: [Troubleshooting](https://www.qodo.ai/blog/technical-faq-and-troubleshooting/)
*   Support:  support@qodo.ai