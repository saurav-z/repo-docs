<div align="center">
  <a href="https://github.com/qodo-ai/pr-agent">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
      <source media="(prefers-color-scheme: light)" srcset="https://www.qodo.ai/wp-content/uploads/2025/02/PR-Agent-Purple-2.png">
      <img src="https://codium.ai/images/pr_agent/logo-light.png" alt="PR-Agent Logo" width="330">
    </picture>
  </a>
  <br/>
  <a href="https://github.com/qodo-ai/pr-agent">PR-Agent: Supercharge your pull request workflow with AI-powered code review, suggestions, and automation.</a>
  <br/>
  <a href="https://qodo-merge-docs.qodo.ai/installation/">Installation Guide</a> |
  <a href="https://qodo-merge-docs.qodo.ai/usage-guide/">Usage Guide</a> |
  <a href="https://qodo-merge-docs.qodo.ai/tools/">Tools Guide</a> |
  <a href="https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/">Qodo Merge 💎</a>
</div>

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-violet)](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl)
[![Pro App](https://img.shields.io/badge/Pro-App-blue)](https://github.com/apps/qodo-merge-pro/)
[![Open Source App](https://img.shields.io/badge/OpenSource-App-red)](https://github.com/apps/qodo-merge-pro-for-open-source/)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=purple)](https://discord.com/invite/SgSxuQ65GF)
<a href="https://github.com/Codium-ai/pr-agent/commits/main">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Codium-ai/pr-agent/main?style=for-the-badge" height="20">
</a>

## Key Features & Benefits

PR-Agent streamlines your pull request process with powerful AI-driven tools, improving code quality and accelerating your workflow.  Key features include:

*   **Automated Code Reviews:** Get instant feedback on your code with AI-powered reviews that highlight potential issues and suggest improvements.
*   **Intelligent Suggestions:**  Receive context-aware suggestions for code enhancements, based on best practices and potential problems.
*   **Automated Tasks:** Automate common PR tasks like generating descriptions, updating changelogs, and more.
*   **Cross-Platform Support:** Works seamlessly with GitHub, GitLab, Bitbucket, and Azure DevOps.
*   **Flexible Deployment:** Use the CLI, GitHub Action, webhooks, or a local IDE post-commit, all with support for multiple models (GPT, Claude, etc).
*   **Qodo Merge Integration:**  Leverage the advanced features of [Qodo Merge](https://www.qodo.ai/pricing/), including a free tier for open-source projects and priority support.
*   **Customizable Tools:** Tailor PR-Agent's behavior with configuration files and custom prompts.
*   **Advanced Core Abilities:**  Includes PR compression, dynamic context, multiple model support, and interactive actions.

Explore the power of AI in your PR process!

## Getting Started

Get up and running with PR-Agent in minutes!

### Try it Instantly

Test PR-Agent on any public GitHub repository by commenting `@CodiumAI-Agent /improve`

### Installation Options

*   **GitHub Action:** Integrate automated PR reviews into your workflow with a simple workflow file.  See the [GitHub Action setup guide](https://qodo-merge-docs.qodo.ai/installation/github/#run-as-a-github-action).

*   **Other Platforms:**
    *   [GitLab webhook setup](https://qodo-merge-docs.qodo.ai/installation/gitlab/)
    *   [BitBucket app installation](https://qodo-merge-docs.qodo.ai/installation/bitbucket/)
    *   [Azure DevOps setup](https://qodo-merge-docs.qodo.ai/installation/azure/)

*   **CLI Usage:** Run PR-Agent locally on your repository using the command line. See the [Local CLI setup guide](https://qodo-merge-docs.qodo.ai/usage-guide/automations_and_usage/#local-repo-cli).

### Qodo Merge 💎

For a fully-managed, feature-rich experience, check out [Qodo Merge](https://www.qodo.ai/pricing/):

*   **FREE for Open Source:**  Enjoy full features at no cost for public repositories.
*   **Easy Setup:** Simple installation with Qodo Merge app for GitHub/GitLab/BitBucket.
*   **Advanced Features:**  Prioritized support and access to unique tools like auto-approve, compliance checks, and more.
*   [Intro and Installation guide](https://qodo-merge-docs.qodo.ai/installation/qodo_merge/)
*   [Plans & Pricing](https://www.qodo.ai/pricing/)

## News and Updates

Stay up-to-date with the latest PR-Agent developments:

*   **Jul 1, 2025:**  Receive automatic feedback from Qodo Merge in your local IDE after each commit.  Learn more [here](https://github.com/qodo-ai/agents/tree/main/agents/qodo-merge-post-commit).
*   **Jun 21, 2025:**  v0.30 was [released](https://github.com/qodo-ai/pr-agent/releases)
*   **Jun 3, 2025:**  Qodo Merge now offers a simplified free tier 💎.  Organizations can use Qodo Merge at no cost with a monthly limit of 75 PR reviews per organization.
*   **Apr 30, 2025:**  New feature: Chat on code suggestions in the `/improve` tool for Qodo Merge 💎.  Read more [here](https://qodo-merge-docs.qodo.ai/tools/improve/#chat-on-code-suggestions).
*   **Apr 16, 2025:**  New tool for Qodo Merge 💎 - `/scan_repo_discussions`.  Read more [here](https://qodo-merge-docs.qodo.ai/tools/scan_repo_discussions/).

## Why Use PR-Agent?

PR-Agent stands out by emphasizing practical, efficient, and affordable PR management. It provides:

*   **Fast Results:**  Each tool uses a single LLM call, ensuring quick feedback (around 30 seconds).
*   **PR Compression:** Effectively handles both short and long pull requests.
*   **Modular Tools:** Customizable tools with a JSON prompting strategy.  Easy to add or modify categories.
*   **Broad Compatibility:**  Supports multiple git providers and models.

## See It in Action

Experience the power of PR-Agent firsthand with these examples:

*   **[/describe](https://github.com/Codium-ai/pr-agent/pull/530)**
    <div align="center">
        <img src="https://www.codium.ai/images/pr_agent/describe_new_short_main.png" width="512">
    </div>
    <hr>

*   **[/review](https://github.com/Codium-ai/pr-agent/pull/732#issuecomment-1975099151)**
    <div align="center">
        <kbd>
        <img src="https://www.codium.ai/images/pr_agent/review_new_short_main.png" width="512">
        </kbd>
    </div>
    <hr>

*   **[/improve](https://github.com/Codium-ai/pr-agent/pull/732#issuecomment-1975099159)**
    <div align="center">
        <kbd>
        <img src="https://www.codium.ai/images/pr_agent/improve_new_short_main.png" width="512">
        </kbd>
    </div>

## How It Works

[Diagram showing PR-Agent tools and their flow](https://www.qodo.ai/images/pr_agent/diagram-v0.9.png)

Learn more about the [PR Compression strategy](https://qodo-merge-docs.qodo.ai/core-abilities/#pr-compression-strategy).

## Data Privacy

PR-Agent is designed with your privacy in mind.

### Self-hosted PR-Agent

*   When using your OpenAI API key, you are subject to OpenAI's privacy policy: [https://openai.com/enterprise-privacy](https://openai.com/enterprise-privacy)

### Qodo-hosted Qodo Merge 💎

*   Qodo Merge (hosted by Qodo) does not store or use your data for model training.
*   Qodo Merge uses an OpenAI account with zero data retention.
*   Qodo Merge utilizes Qodo's proprietary models for certain clients (you will be notified).
*   Qodo Merge is only active when you invoke it, analyzing only the data relevant to your command and pull request.

### Qodo Merge Chrome extension

*   The [Qodo Merge Chrome extension](https://chromewebstore.google.com/detail/qodo-merge-ai-powered-cod/ephlnjeghhogofkifjloamocljapahnl) modifies the visual appearance of GitHub PR screens, but does not transmit any user code or repo information except when a PR-Agent tool is used, and is compliant with standard Qodo-Merge privacy policy.

## Contributing

Contribute to the project! Review the [Contributing Guide](https://github.com/qodo-ai/pr-agent/blob/b09eec265ef7d36c232063f76553efb6b53979ff/CONTRIBUTING.md) to get started.

## Links

*   **[GitHub Repository](https://github.com/qodo-ai/pr-agent)**
*   [Discord Community](https://discord.com/invite/SgSxuQ65GF)
*   [Qodo Site](https://www.qodo.ai/)
*   [Blog](https://www.qodo.ai/blog/)
*   [Troubleshooting](https://www.qodo.ai/blog/technical-faq-and-troubleshooting/)
*   [Support](mailto:support@qodo.ai)