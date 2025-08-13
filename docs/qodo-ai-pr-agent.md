<div align="center">
  <a href="https://github.com/qodo-ai/pr-agent">
    <img src="https://codium.ai/images/pr_agent/logo-light.png" alt="PR-Agent Logo" width="330">
  </a>
  <br/>
  <p><b>Supercharge your pull requests with AI!</b> PR-Agent helps you efficiently review, improve, and manage your code changes.</p>
  <br/>
  <a href="https://qodo-merge-docs.qodo.ai/installation/">Installation Guide</a> |
  <a href="https://qodo-merge-docs.qodo.ai/usage-guide/">Usage Guide</a> |
  <a href="https://qodo-merge-docs.qodo.ai/tools/">Tools Guide</a> |
  <a href="https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/">Qodo Merge 💎</a>
</div>

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Qodo Merge Pro](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![Open Source](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
[![Last Commit](https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge)](https://github.com/Codium-ai/pr-agent/commits/main)

## Key Features of PR-Agent

*   **AI-Powered Review:** Get intelligent feedback on your code changes.
*   **Automated Suggestions:** Receive AI-generated suggestions for improvements.
*   **Multi-Platform Support:** Works with GitHub, GitLab, BitBucket, and Azure DevOps.
*   **Multiple Integration Methods:** Use via CLI, GitHub Actions, or webhooks.
*   **Customizable Tools:** Easily configure and extend PR-Agent's functionality.
*   **Qodo Merge Integration:**  Leverage advanced features like code analysis, CI feedback, and more (💎 features) within the hosted Qodo Merge platform.
*   **Free for Open Source:** Access all features at no cost for public repositories using Qodo Merge.

**[Explore the PR-Agent Repository](https://github.com/qodo-ai/pr-agent)**

## Getting Started

PR-Agent offers several ways to get started:

*   **Instant Trial:** Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve`.

*   **GitHub Action:** Integrate automated PR reviews with a simple workflow file: [GitHub Action setup guide](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action)

*   **Other Platforms:**
    *   [GitLab webhook setup](https://qodo-merge-docs.qodo.ai/installation/gitlab/)
    *   [BitBucket app installation](https://qodo-merge-docs.qodo.ai/installation/bitbucket/)
    *   [Azure DevOps setup](https://qodo-merge-docs.qodo.ai/installation/azure/)

*   **CLI Usage:** Run PR-Agent locally: [Local CLI setup guide](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli)

*   **Qodo Merge as post-commit:** Receive automatic feedback from Qodo Merge on your local IDE after each [commit](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

### Qodo Merge 💎

[Qodo Merge](https://www.qodo.ai/pricing/) is a hosted platform offering enhanced features:

*   **FREE for Open Source:** Full features, zero cost for public repos.
*   [Intro and Installation guide](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/)
*   [Plans & Pricing](https://www.qodo.ai/pricing/)
*   **Zero-setup hosted solution with advanced features and priority support**

### Qodo Merge as a Post-commit in Your Local IDE
You can receive automatic feedback from Qodo Merge on your local IDE after each [commit](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)


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

## Why Use PR-Agent?

PR-Agent provides these key advantages:

*   **Fast and Affordable:** Each tool uses a single LLM call for quick (~30 seconds) and cost-effective results.
*   **Effective PR Handling:** Employs a PR Compression strategy for both short and long PRs.
*   **Modular and Customizable:** JSON prompting enables easy configuration of tools.
*   **Broad Compatibility:** Supports various Git providers, integration methods, and LLM models.

## Features

PR-Agent and Qodo Merge offer extensive pull request functionalities:

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
- 💎 means this feature is available only in [Qodo Merge](https://www.qodo.ai/pricing/)

[//]: # (- Support for additional git providers is described in [here]&#40;./docs/Full_environments.md&#41;)
___

## See It in Action

**Describe Tool:**

<div align="center">
  <img src="https://www.codium.ai/images/pr_agent/describe_new_short_main.png" width="512" alt="Describe Tool Example">
</div>
<hr>

**Review Tool:**

<div align="center">
  <kbd>
    <img src="https://www.codium.ai/images/pr_agent/review_new_short_main.png" width="512" alt="Review Tool Example">
  </kbd>
</div>
<hr>

**Improve Tool:**

<div align="center">
  <kbd>
    <img src="https://www.codium.ai/images/pr_agent/improve_new_short_main.png" width="512" alt="Improve Tool Example">
  </kbd>
</div>

<hr>

## Try It Now

To experience PR-Agent's capabilities, simply mention `@CodiumAI-Agent` followed by a command (e.g., `/review`) in any PR comment within your public GitHub repository.

---

## How It Works

The following diagram illustrates PR-Agent tools and their flow:

![PR-Agent Tools](https://www.qodo.ai/images/pr_agent/diagram-v0.9.png)

Check out the [PR Compression strategy](https://qodo-merge-docs.qodo.ai/core-abilities/#pr-compression-strategy) page for more details on how we convert a code diff to a manageable LLM prompt

## Data Privacy

### Self-hosted PR-Agent

*   If you host PR-Agent with your OpenAI API key, it is between you and OpenAI. You can read their API data privacy policy here:
    [https://openai.com/enterprise-privacy](https://openai.com/enterprise-privacy)

### Qodo-hosted Qodo Merge 💎

*   When using Qodo Merge 💎, hosted by Qodo, we will not store any of your data, nor will we use it for training. You will also benefit from an OpenAI account with zero data retention.

*   For certain clients, Qodo-hosted Qodo Merge will use Qodo’s proprietary models — if this is the case, you will be notified.

*   No passive collection of Code and Pull Requests’ data — Qodo Merge will be active only when you invoke it, and it will then extract and analyze only data relevant to the executed command and queried pull request.

### Qodo Merge Chrome extension

*   The [Qodo Merge Chrome extension](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl) serves solely to modify the visual appearance of a GitHub PR screen. It does not transmit any user's repo or pull request code. Code is only sent for processing when a user submits a GitHub comment that activates a PR-Agent tool, in accordance with the standard privacy policy of Qodo-Merge.

## Contributing

Contribute to PR-Agent by reviewing the [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md).

## Links

*   **Discord:** [https://discord.com/invite/SgSxuQ65GF](https://discord.com/invite/SgSxuQ65GF)
*   **Qodo Site:** [https://www.qodo.ai/](https://www.qodo.ai/)
*   **Blog:** [https://www.qodo.ai/blog/](https://www.qodo.ai/blog/)
*   **Troubleshooting:** [https://www.qodo.ai/blog/technical-faq-and-troubleshooting/](https://www.qodo.ai/blog/technical-faq-and-troubleshooting/)
*   **Support:** support@qodo.ai