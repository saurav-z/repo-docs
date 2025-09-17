<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Evaluate, Monitor, and Optimize Your Self-Learning Agents

**Judgeval empowers you to build, test, and refine your autonomous agents with data-driven insights.**

[**View the Judgeval Repository on GitHub**](https://github.com/JudgmentLabs/judgeval)

[Docs](https://docs.judgmentlabs.ai/)  •  [Judgment Cloud](https://app.judgmentlabs.ai/register)  • [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)  • [Landing Page](https://judgmentlabs.ai/)

 [Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) • [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) • [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

Judgeval is an open-source toolkit for evaluating and monitoring autonomous agents, providing the essential data and signals needed for continuous learning and improvement.

## Key Features

*   **Evals:** Build custom evaluators using LLM-as-a-judge, manual labeling, or code-based solutions. Integrate with our metric-tracking infrastructure for comprehensive evaluation.
    *   Unit-testing
    *   A/B testing
    *   Online guardrails
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

*   **Monitoring:** Get real-time alerts for agent failures and identify performance regressions with custom hooks and Slack integration.
    *   Identifying degradation early
    *   Visualizing performance trends across agent versions and time
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

*   **Datasets:** Export agent interactions to datasets for in-depth analysis and optimization. Move data to/from Parquet, S3, etc. Run evals on datasets to A/B test different agent configurations.
    *   Agent environment interaction data for optimization
    *   Scaled analysis for A/B tests
    <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🎬 Judgeval in Action

**[Multi-Agent System](https://github.com/JudgmentLabs/judgment-cookbook/tree/main/cookbooks/agents/multi-agent) with complete observability:** (1) A multi-agent system spawns agents to research topics on the internet. (2) With just **3 lines of code**, Judgeval captures all environment responses across all agent tool calls for monitoring. (3) After completion, (4) export all interaction data to enable further environment-specific learning and optimization.

<table style="width: 100%; max-width: 800px; table-layout: fixed;">
<tr>
<td align="center" style="padding: 8px; width: 50%;">
  <img src="assets/agent.gif" alt="Agent Demo" style="width: 100%; max-width: 350px; height: auto;" />
  <br><strong>🤖 Agents Running</strong>
</td>
<td align="center" style="padding: 8px; width: 50%;">
  <img src="assets/trace.gif" alt="Capturing Environment Data Demo" style="width: 100%; max-width: 350px; height: auto;" />
  <br><strong>📊 Capturing Environment Data </strong>
</td>
</tr>
<tr>
<td align="center" style="padding: 8px; width: 50%;">
  <img src="assets/document.gif" alt="Agent Completed Demo" style="width: 100%; max-width: 350px; height: auto;" />
  <br><strong>✅ Agents Completed Running</strong>
</td>
<td align="center" style="padding: 8px; width: 50%;">
  <img src="assets/data.gif" alt="Data Export Demo" style="width: 100%; max-width: 350px; height: auto;" />
  <br><strong>📤 Exporting Agent Environment Data</strong>
</td>
</tr>

</table>

## 🛠️ Installation

Get started by installing the Judgeval SDK:

```bash
pip install judgeval
```

Set your API keys to connect to the [Judgment Platform](https://app.judgmentlabs.ai/register).

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have API keys? [Create an account](https://app.judgmentlabs.ai/register) on the platform!**

## 🏢 Self-Hosting

Take full control by self-hosting Judgeval on your own infrastructure.

### Key Benefits
*   Deploy on your own AWS account
*   Store data in your own Supabase instance
*   Access through your custom domain

### Getting Started

1.  Follow the [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend.

## 📚 Cookbooks

Explore example applications in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook) repository.  We're happy to feature your contributions - create a PR or message us on [Discord](https://discord.gg/tGVFf8UBUY).

## 💻 Development with Cursor

Enhance your coding experience in Cursor with Judgeval integration. The [Cursor rules file](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) provides key context for your coding assistant to work effectively with Judgment features.

## ⭐ Star Us on GitHub

Show your support! If you find Judgeval valuable, give us a star on GitHub!

## ❤️ Contributing

We welcome contributions!

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
*   Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls)
*   Share your Judgeval experience!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).