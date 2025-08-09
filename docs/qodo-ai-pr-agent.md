<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
    <source media="(prefers-color-scheme: light)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
    <img src="https://codium.ai/images/pr_agent/logo-light.png" alt="PR-Agent Logo" width="330">
  </picture>
  <br/>
  **Revolutionize your pull request workflow with PR-Agent, the AI-powered assistant designed to streamline code reviews and boost developer productivity.**

  [Installation Guide](https://qodo-merge-docs.qodo.ai/installation/) |
  [Usage Guide](https://qodo-merge-docs.qodo.ai/usage-guide/) |
  [Tools Guide](https://qodo-merge-docs.qodo.ai/tools/) |
  [Qodo Merge](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) 💎
</div>

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Pro App](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![Open Source App](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
<a href="https://github.com/Codium-ai/pr-agent/commits/main">
  <img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge" height="20">
</a>

## Table of Contents

- [Key Features](#key-features)
- [Why Use PR-Agent?](#why-use-pr-agent)
- [Getting Started](#getting-started)
- [News and Updates](#news-and-updates)
- [See PR-Agent in Action](#see-it-in-action)
- [Try It Now](#try-it-now)
- [Qodo Merge 💎](#qodo-merge-)
- [How It Works](#how-it-works)
- [Data Privacy](#data-privacy)
- [Contributing](#contributing)
- [Links](#links)
- [Original Repo](#original-repo)

## Key Features

PR-Agent and Qodo Merge provide a suite of AI-powered tools to enhance your pull request workflow. Key features include:

*   **Comprehensive Code Review:** Automates code analysis with tools like `/review`, `/improve`, and `/analyze`.
*   **Automated Documentation:**  Generates and updates documentation with tools like `/describe` and `/add_documentation`.
*   **Multi-Platform Support:**  Integrates seamlessly with GitHub, GitLab, Bitbucket, and Azure DevOps.
*   **Flexible Deployment:**  Offers CLI, GitHub Action, GitHub App, and Docker options.
*   **Advanced Capabilities:** Includes features like PR compression, dynamic context, and integration with ticket systems.

  <details>
  <summary>
  **Complete Feature List:**
  </summary>

<div style="text-align:left;">

PR-Agent and Qodo Merge offer comprehensive pull request functionalities integrated with various git providers:

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

  </div>
  </details>

## Why Use PR-Agent?

PR-Agent offers several advantages over traditional code review methods:

*   **Efficiency:**  Get rapid feedback and suggestions, reducing review times.
*   **Cost-Effectiveness:**  Leverages single LLM calls for fast and affordable results.
*   **Customization:** Easily configure and extend tools through JSON prompting.
*   **Comprehensive Support:** Works with popular Git providers and multiple LLM models.

## Getting Started

PR-Agent can be integrated into your workflow in multiple ways:

*   **Try It Instantly:** Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve`.
*   **GitHub Action:** Automate PR reviews using a simple workflow file. See the [GitHub Action setup guide](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action).
*   **Other Platforms:** Integration guides are available for [GitLab](https://qodo-merge-docs.qodo.ai/installation/gitlab/), [BitBucket](https://qodo-merge-docs.qodo.ai/installation/bitbucket/), and [Azure DevOps](https://qodo-merge-docs.qodo.ai/installation/azure/).
*   **CLI Usage:** Run PR-Agent locally with the [Local CLI setup guide](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli).
*   **Qodo Merge IDE Integration:** Integrate Qodo Merge as a post-commit process in your local IDE (see [here](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)).

## News and Updates

### Aug 8, 2025

Added full support for GPT-5 models. View the [benchmark results](https://qodo-merge-docs.qodo.ai/pr_benchmark/#pr-benchmark-results) for details on the performance of GPT-5 models in PR-Agent.

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

## See PR-Agent in Action

Explore how PR-Agent streamlines code review through these examples:

*   **/describe**: [See Example](https://github.com/Codium-ai/pr-agent/pull/530)
    <div align="center">
        <img src="https://www.codium.ai/images/pr_agent/describe_new_short_main.png" width="512">
    </div>
    <hr>

*   **/review**: [See Example](https://github.com/Codium-ai/pr-agent/pull/732#issuecomment-1975099151)
    <div align="center">
    <kbd>
        <img src="https://www.codium.ai/images/pr_agent/review_new_short_main.png" width="512">
    </kbd>
    </div>
    <hr>

*   **/improve**: [See Example](https://github.com/Codium-ai/pr-agent/pull/732#issuecomment-1975099159)
    <div align="center">
    <kbd>
        <img src="https://www.codium.ai/images/pr_agent/improve_new_short_main.png" width="512">
    </kbd>
    </div>
    <hr>

## Try It Now

To get started with PR-Agent, simply mention `@CodiumAI-Agent` in a comment on your public GitHub repository's pull request along with a command, such as `/review`.  The bot will then provide feedback based on your specified command.

**Note:** This is a promotional bot and may not be available on private repositories.

---

## Qodo Merge 💎

[Qodo Merge](https://www.qodo.ai/pricing/) offers a fully managed, hosted version of PR-Agent, providing several benefits:

1.  **Fully Managed:** Qodo handles hosting, model updates, and more, simplifying setup.
2.  **Enhanced Privacy:** No data storage or model training using your data, employing an OpenAI account with zero data retention.
3.  **Priority Support:** Qodo Merge users receive dedicated support and feature request options.
4.  **Advanced Features:**  Emphasis on customization and static code analysis.

See [here](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) for a list of features available in Qodo Merge.

## How It Works

The following diagram illustrates the PR-Agent workflow and tools:

![PR-Agent Tools](https://www.qodo.ai/images/pr_agent/diagram-v0.9.png)

Learn more about the [PR Compression strategy](https://qodo-merge-docs.qodo.ai/core-abilities/#pr-compression-strategy) to understand how code diffs are converted for LLM prompts.

## Data Privacy

### Self-hosted PR-Agent

*   If you host PR-Agent with your OpenAI API key, you are responsible for the data privacy. See OpenAI's API data privacy policy [here](https://openai.com/enterprise-privacy).

### Qodo-hosted Qodo Merge 💎

*   Qodo Merge 💎 does not store or use your data for training and utilizes an OpenAI account with zero data retention.
*   For certain clients, Qodo-hosted Qodo Merge will use Qodo’s proprietary models — if this is the case, you will be notified.
*   Qodo Merge analyzes only the data relevant to the executed command on invocation.

### Qodo Merge Chrome extension

*   The [Qodo Merge Chrome extension](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl) modifies the GitHub PR screen's visual appearance, without transmitting user's repo or PR code. Code is only sent for processing when a user submits a GitHub comment that activates a PR-Agent tool, in accordance with the standard privacy policy of Qodo-Merge.

## Contributing

Contribute to the project by reading our [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md).

## Links

*   **Discord:** https://discord.com/invite/SgSxuQ65GF
*   **Qodo Site:** https://www.qodo.ai/
*   **Blog:** https://www.qodo.ai/blog/
*   **Troubleshooting:** https://www.qodo.ai/blog/technical-faq-and-troubleshooting/
*   **Support:** support@qodo.ai

## Original Repo

[PR-Agent GitHub Repository](https://github.com/qodo-ai/pr-agent)