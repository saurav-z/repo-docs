<div align="center">
  <a href="https://github.com/qodo-ai/pr-agent">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
      <source media="(prefers-color-scheme: light)" srcset="https://codium.ai/images/pr_agent/logo-light.png">
      <img src="https://codium.ai/images/pr_agent/logo-light.png" alt="PR-Agent Logo" width="330">
    </picture>
  </a>
  <br/>
  <p>
    <b>PR-Agent: Revolutionizing Pull Request Management with AI-Powered Automation.</b>
  </p>
  [Installation Guide](https://qodo-merge-docs.qodo.ai/installation/) |
  [Usage Guide](https://qodo-merge-docs.qodo.ai/usage-guide/) |
  [Tools Guide](https://qodo-merge-docs.qodo.ai/tools/) |
  [Qodo Merge Pro](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) 💎
</div>

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Qodo Merge Pro](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![Open Source](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
[![Last Commit](https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge)](https://github.com/qodo-ai/pr-agent/commits/main)

## Table of Contents

*   [Key Features](#key-features)
*   [Getting Started](#getting-started)
*   [Why Use PR-Agent?](#why-use-pr-agent)
*   [News and Updates](#news-and-updates)
*   [See It in Action](#see-it-in-action)
*   [Try It Now](#try-it-now)
*   [Qodo Merge 💎](#qodo-merge-)
*   [How It Works](#how-it-works)
*   [Data Privacy](#data-privacy)
*   [Contributing](#contributing)
*   [Links](#links)

## Key Features

*   **Automated PR Reviews:** Get instant feedback and suggestions on your pull requests.
*   **Multi-Platform Support:** Compatible with GitHub, GitLab, Bitbucket, and Azure DevOps.
*   **AI-Powered Tools:** Utilize tools for describing, reviewing, improving, and more.
*   **Local CLI & Integrations:** Use via CLI, GitHub Actions, webhooks, and IDE integrations.
*   **Qodo Merge Pro:** Access advanced features with a hosted, managed solution.
*   **Multiple Models:** Utilize a wide range of LLMs from OpenAI to Claude, and more.

## Getting Started

### Try it Instantly

Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve`.

### GitHub Action

Add automated PR reviews to your repository with a simple workflow file using [GitHub Action setup guide](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action).

#### Other Platforms

*   [GitLab webhook setup](https://qodo-merge-docs.qodo.ai/installation/gitlab/)
*   [BitBucket app installation](https://qodo-merge-docs.qodo.ai/installation/bitbucket/)
*   [Azure DevOps setup](https://qodo-merge-docs.qodo.ai/installation/azure/)

### CLI Usage

Run PR-Agent locally on your repository via command line: [Local CLI setup guide](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli)

### Qodo Merge as post-commit in your local IDE

See [here](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

### Discover Qodo Merge 💎

Zero-setup hosted solution with advanced features and priority support

*   **[FREE for Open Source](https://github.com/marketplace/qodo-merge-pro-for-open-source)**: Full features, zero cost for public repos
*   [Intro and Installation guide](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/)
*   [Plans & Pricing](https://www.qodo.ai/pricing/)

### Qodo Merge as a Post-commit in Your Local IDE

You can receive automatic feedback from Qodo Merge on your local IDE after each [commit](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit)

## Why Use PR-Agent?

PR-Agent offers a fast, affordable, and customizable solution for streamlining your pull request workflow.  Key advantages include:

*   **Fast and Affordable:** Each tool relies on a single LLM call for quick and cost-effective responses (approx. 30 seconds).
*   **PR Compression:** PR-Agent efficiently handles both short and long pull requests.
*   **Modular Tools:** Highly customizable tools using a JSON prompting strategy.  Easy to add new categories and features.
*   **Wide Compatibility:** Supports GitHub, GitLab, BitBucket, and more, with various integration options.
*   **Multiple Models:** Choice of AI models including GPT, Claude, and Deepseek.

## News and Updates

#### Aug 8, 2025

Added full support for GPT-5 models. View the [benchmark results](https://qodo-merge-docs.qodo.ai/pr_benchmark/#pr-benchmark-results) for details on the performance of GPT-5 models in PR-Agent.

#### Jul 1, 2025

You can now receive automatic feedback from Qodo Merge in your local IDE after each commit. Read more about it [here](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit).

#### Jun 21, 2025

v0.30 was [released](https://github.com/qodo-ai/pr-agent/releases)

#### Jun 3, 2025

Qodo Merge now offers a simplified free tier 💎. Organizations can use Qodo Merge at no cost, with a [monthly limit](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/#cloud-users) of 75 PR reviews per organization.

#### Apr 30, 2025

A new feature is now available in the `/improve` tool for Qodo Merge 💎 - Chat on code suggestions.
<img width="512" alt="image" src="https://codium.ai/images/pr_agent/improve_chat_on_code_suggestions_ask.png" />
Read more about it [here](https://qodo-merge-docs.qodo.ai/tools/improve/#chat-on-code-suggestions).

#### Apr 16, 2025

New tool for Qodo Merge 💎 - `/scan_repo_discussions`.
<img width="635" alt="image" src="https://codium.ai/images/pr_agent/scan_repo_discussions_2.png" />
Read more about it [here](https://qodo-merge-docs.qodo.ai/tools/scan_repo_discussions/).

## See It in Action

<h4><a href="https://github.com/Codium-ai/pr-agent/pull/530">/describe</a></h4>
<div align="center">
<p float="center">
<img src="https://www.codium.ai/images/pr_agent/describe_new_short_main.png" width="512">
</p>
</div>
<hr>

<h4><a href="https://github.com/Codium-ai/pr-agent/pull/732#issuecomment-1975099151">/review</a></h4>
<div align="center">
<p float="center">
<kbd>
<img src="https://www.codium.ai/images/pr_agent/review_new_short_main.png" width="512">
</kbd>
</p>
</div>
<hr>

<h4><a href="https://github.com/Codium-ai/pr-agent/pull/732#issuecomment-1975099159">/improve</a></h4>
<div align="center">
<p float="center">
<kbd>
<img src="https://www.codium.ai/images/pr_agent/improve_new_short_main.png" width="512">
</kbd>
</p>
</div>

## Try It Now

Instantly test the GPT-5 powered PR-Agent on your public GitHub repository. Just mention `@CodiumAI-Agent` with a command like `/review` in a PR comment.

Note: This is a promotional bot and cannot edit your repo or be used on private repositories.

---

## Qodo Merge 💎

[Qodo Merge](https://www.qodo.ai/pricing/) is a hosted, fully-managed version of PR-Agent offered by Qodo. Benefits include:

1.  **Fully Managed:** Hosted, updated, and maintained by Qodo. Installation is easy.
2.  **Enhanced Privacy:** No data storage or model training with your data. Utilizes OpenAI with zero data retention.
3.  **Priority Support:** Qodo Merge users receive prioritized support and can request new features.
4.  **Extra Features:** More customization, static code analysis, and LLM integration. See [here](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) for Qodo Merge features.

## How It Works

The diagram below illustrates PR-Agent's tools and their workflow:

![PR-Agent Tools](https://www.qodo.ai/images/pr_agent/diagram-v0.9.png)

For details on how PR-Agent converts code diffs, see [PR Compression strategy](https://qodo-merge-docs.qodo.ai/core-abilities/#pr-compression-strategy).

## Data Privacy

### Self-hosted PR-Agent

*   If you self-host PR-Agent using your OpenAI API key, you are subject to OpenAI's privacy policy: [https://openai.com/enterprise-privacy](https://openai.com/enterprise-privacy)

### Qodo-hosted Qodo Merge 💎

*   Qodo Merge 💎 hosted by Qodo does not store or use your data for training. It also utilizes OpenAI with zero data retention.
*   Qodo-hosted Qodo Merge may use Qodo’s proprietary models (you will be notified).
*   Qodo Merge is only active when invoked, analyzing only the data relevant to the command and pull request.

### Qodo Merge Chrome extension

*   The [Qodo Merge Chrome extension](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl) only modifies the visual appearance of the GitHub PR screen and does not transmit any user's repo or pull request code. Code is only sent for processing when a user submits a GitHub comment that activates a PR-Agent tool, in accordance with the standard privacy policy of Qodo-Merge.

## Contributing

Contribute to the project by reading our [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md).

## Links

*   **GitHub Repository:** [https://github.com/qodo-ai/pr-agent](https://github.com/qodo-ai/pr-agent)
*   Discord community: https://discord.com/invite/SgSxuQ65GF
*   Qodo site: https://www.qodo.ai/
*   Blog: https://www.qodo.ai/blog/
*   Troubleshooting: https://www.qodo.ai/blog/technical-faq-and-troubleshooting/
*   Support: support@qodo.ai