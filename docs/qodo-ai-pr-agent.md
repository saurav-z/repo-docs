<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
  <source media="(prefers-color-scheme: light)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
  <img src="https://codium.ai/images/pr_agent/logo-light.png" alt="PR-Agent Logo" width="330">
</picture>

</div>

## PR-Agent: AI-Powered Pull Request Review and Automation

**PR-Agent, available on [GitHub](https://github.com/qodo-ai/pr-agent), streamlines your pull request workflow with AI-driven feedback, suggestions, and automation, enhancing code quality and developer productivity.**

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Pro App](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![Open Source App](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
<a href="https://github.com/Codium-ai/pr-agent/commits/main">
<img alt="GitHub" src="https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge" height="20">
</a>

### Key Features:

*   **Automated Code Review:** Get instant feedback on your code changes.
*   **AI-Powered Suggestions:** Receive intelligent recommendations for improvement.
*   **Multiple Git Provider Support:** Works seamlessly with GitHub, GitLab, BitBucket, and Azure DevOps.
*   **Flexible Integration:** Use it via CLI, GitHub Actions, or as a Chrome extension.
*   **Qodo Merge Integration:** Access advanced features and priority support through the hosted Qodo Merge platform.
*   **Customizable Tools:** Configure the `/review` tool and create custom prompts for tailored feedback.
*   **Efficient and Affordable:** Get quick results with a single LLM call per tool, ensuring fast and cost-effective use.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Features Overview](#features)
*   [Why Use PR-Agent?](#why-use-pr-agent)
*   [See It in Action](#see-it-in-action)
*   [Try It Now](#try-it-now)
*   [Qodo Merge 💎](#qodo-merge-)
*   [How It Works](#how-it-works)
*   [Data Privacy](#data-privacy)
*   [Contributing](#contributing)
*   [Links](#links)

## Getting Started

PR-Agent is designed for easy integration into your workflow.  Choose your preferred method:

### Try It Instantly

Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve`.

### GitHub Action

Integrate automated PR reviews into your repository with a simple workflow file using our [GitHub Action setup guide](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action).

### Other Platforms

*   [GitLab webhook setup](https://qodo-merge-docs.qodo.ai/installation/gitlab/)
*   [BitBucket app installation](https://qodo-merge-docs.qodo.ai/installation/bitbucket/)
*   [Azure DevOps setup](https://qodo-merge-docs.qodo.ai/installation/azure/)

### CLI Usage

Run PR-Agent locally on your repository via command line: [Local CLI setup guide](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli)

### Qodo Merge as post-commit in your local IDE

See [here](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

### Discover Qodo Merge 💎

Unlock advanced features with Qodo Merge, a zero-setup hosted solution:

*   **[FREE for Open Source](https://github.com/marketplace/qodo-merge-pro-for-open-source)**: Access full features at no cost for public repositories.
*   [Intro and Installation guide](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/)
*   [Plans & Pricing](https://www.qodo.ai/pricing/)

### Qodo Merge as a Post-commit in Your Local IDE

Receive automatic feedback from Qodo Merge on your local IDE after each [commit](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

## Features Overview

PR-Agent and Qodo Merge provide a robust suite of tools for pull request management, supporting various git providers:

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
|                                                         |                                                                                                                     |        |        |           |              |       |

*💎 denotes features available in Qodo Merge.*

## Why Use PR-Agent?

PR-Agent distinguishes itself with:

*   **Practical Use:** Tools are designed for real-world team usage, providing quick and affordable results with a single LLM call.
*   **PR Compression Strategy:** Effectively handles both short and long pull requests.
*   **Modular and Customizable:** JSON prompting allows for flexible tool configuration.
*   **Multi-Platform Support:** Compatible with GitHub, GitLab, BitBucket, and various deployment methods (CLI, GitHub Action, Docker, etc.).
*   **Multiple Model Support:**  Offers choices of GPT, Claude, Deepseek, and more.

## See It in Action

*   **/describe:**

    <div align="center">
    <p float="center">
    <img src="https://www.codium.ai/images/pr_agent/describe_new_short_main.png" width="512">
    </p>
    </div>
    <hr>
*   **/review:**

    <div align="center">
    <p float="center">
    <kbd>
    <img src="https://www.codium.ai/images/pr_agent/review_new_short_main.png" width="512">
    </kbd>
    </p>
    </div>
    <hr>
*   **/improve:**

    <div align="center">
    <p float="center">
    <kbd>
    <img src="https://www.codium.ai/images/pr_agent/improve_new_short_main.png" width="512">
    </kbd>
    </p>
    </div>

## Try It Now

Use PR-Agent on your public GitHub repository. Simply mention `@CodiumAI-Agent /command` in any PR comment.
For example,  `@CodiumAI-Agent /review` to get an instant PR review.

*Note:* This is a promotional bot, not suitable for private repositories or editing features (e.g., description or labels).

---

## Qodo Merge 💎

Qodo Merge is a fully managed, hosted version of PR-Agent, offering:

1.  **Full Management:** We handle hosting, models, and updates. Installation is easy through the Qodo Merge app for GitHub/GitLab/BitBucket.
2.  **Enhanced Privacy:** No data storage or model training with your data. It uses an OpenAI account with zero data retention.
3.  **Prioritized Support:** Qodo Merge users receive priority support and can request new features.
4.  **Advanced Features:** Offers more customization and static code analysis enhancements.
    Find more features at [Qodo Merge](https://www.qodo.ai/pricing/).

## How It Works

The PR-Agent architecture is illustrated below:

![PR-Agent Tools](https://www.qodo.ai/images/pr_agent/diagram-v0.9.png)

Learn more about our  [PR Compression strategy](https://qodo-merge-docs.qodo.ai/core-abilities/#pr-compression-strategy) to understand how we process code diffs.

## Data Privacy

### Self-hosted PR-Agent

*   Data privacy is between you and OpenAI, see their policy: [OpenAI Enterprise Privacy](https://openai.com/enterprise-privacy)

### Qodo-hosted Qodo Merge 💎

*   No data is stored or used for training. Qodo Merge utilizes an OpenAI account with zero data retention.
*   Qodo Merge might use Qodo's proprietary models (clients will be notified).
*   Qodo Merge is active only when you invoke it, extracting and analyzing data specific to the executed command and pull request.

### Qodo Merge Chrome extension

*   The extension modifies the GitHub PR screen's appearance and doesn't transmit code unless activated by a user comment, adhering to Qodo-Merge's standard privacy policy.

## Contributing

Contribute by following our [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md).

## Links

*   Discord community: https://discord.com/invite/SgSxuQ65GF
*   Qodo site: https://www.qodo.ai/
*   Blog: https://www.qodo.ai/blog/
*   Troubleshooting: https://www.qodo.ai/blog/technical-faq-and-troubleshooting/
*   Support: support@qodo.ai