<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

## Judgeval: Supercharge Your Self-Learning Agents

**Empower your self-learning agents with the data and tools they need to excel.**

[Docs](https://docs.judgmentlabs.ai/)  •  [Judgment Cloud](https://app.judgmentlabs.ai/register)  • [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)  • [Landing Page](https://judgmentlabs.ai/)

[Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) • [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) • [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

Join us! We're hiring and building the future of self-learning agents.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

**Judgeval** is an open-source platform providing powerful tools for evaluating and optimizing autonomous, stateful agents. It captures real-time agent-environment interactions, enabling continuous learning and improvement.

## Key Features

*   **Evaluations:** Build custom evaluators using LLMs, manual labeling, or code-based methods. Connect with our metric-tracking infrastructure for comprehensive analysis.
    *   Unit-testing
    *   A/B testing
    *   Online guardrails
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

*   **Monitoring:** Receive instant Slack alerts for agent failures and integrate custom hooks to address production regressions.
    *   Identifying degradation early
    *   Visualizing performance trends
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

*   **Datasets:** Export agent interactions to datasets for scaled analysis and optimization. Analyze data in Parquet, S3, and other formats. Run evals on datasets to A/B test and enable continuous learning.
    *   Agent environment interaction data for optimization
    *   Scaled analysis for A/B tests
    <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🎬 See Judgeval in Action

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

Get started with Judgeval by installing the SDK using pip:

```bash
pip install judgeval
```

Set your API keys to connect to the [Judgment Platform](https://app.judgmentlabs.ai/register) or your self-hosted instance:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have API keys? [Create an account](https://app.judgmentlabs.ai/register) on the platform!**

## 🏢 Self-Hosting

Take full control of your data with Judgeval's self-hosting capabilities.  Run Judgeval on your own infrastructure.

### Key Benefits
*   Deploy Judgeval on your own AWS account
*   Store data in your own Supabase instance
*   Access Judgeval through your own custom domain

### Getting Started

1.  Follow the detailed [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks & Community

Explore ready-to-use examples and integrations in our [cookbooks](https://github.com/JudgmentLabs/judgment-cookbook).  We welcome contributions!  Share your use cases on [Discord](https://discord.gg/tGVFf8UBUY) or create a pull request.

## 💻 Development with Cursor

Integrate Judgeval seamlessly with Cursor for enhanced coding and LLM workflow development.  Get the necessary context with the [Cursor rules file](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules).

## ⭐ Contribute & Support

Help us build the future of self-learning agents!

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Improve our [documentation](https://docs.judgmentlabs.ai/) and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls).
*   Give us a star on GitHub!

**[Visit our GitHub repository](https://github.com/JudgmentLabs/judgeval) to learn more.**

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is brought to you by [Judgment Labs](https://judgmentlabs.ai/).