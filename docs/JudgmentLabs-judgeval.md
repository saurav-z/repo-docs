<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

<br>
<div style="font-size: 1.5em;">
    Enable self-learning agents with environment data and evals.
</div>

## Empower Your AI Agents with Judgeval: The Open-Source Evaluation and Monitoring Toolkit

**Judgeval** is an open-source toolkit that provides the crucial data and signals your autonomous agents need to thrive, enabling continuous learning and self-improvement.

[**Explore the Judgeval GitHub Repository**](https://github.com/JudgmentLabs/judgeval) | [Docs](https://docs.judgmentlabs.ai/) | [Judgment Cloud](https://app.judgmentlabs.ai/register) | [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) | [Landing Page](https://judgmentlabs.ai/)

[Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) • [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) • [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

Join us! We're hiring to further our mission of empowering self-learning agents.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

## Key Features of Judgeval

Judgeval helps you monitor, evaluate, and optimize your AI agents. Here are some of the key features:

*   **Agent Monitoring:** Real-time tracking and alerting for agent failures. Integrate custom hooks to address regressions.
*   **Evaluation & Testing:** Build custom evaluators using LLMs, manual labeling, or code. Supports unit testing, A/B testing, and online guardrails.
*   **Dataset Creation:** Export agent interactions to datasets for scaled analysis and optimization, and easily move datasets to/from Parquet, S3, etc.
*   **Self-Hosting:** Take complete control with self-hosting capabilities, allowing you to deploy Judgeval on your infrastructure.

### 🎬 See Judgeval in Action

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

## 📋 Table of Contents
- [🛠️ Installation](#️-installation)
- [🏁 Quickstarts](#-quickstarts)
- [✨ Features](#-features)
- [🏢 Self-Hosting](#-self-hosting)
- [📚 Cookbooks](#-cookbooks)
- [💻 Development with Cursor](#-development-with-cursor)

## 🛠️ Installation

Install the Judgeval SDK using pip:

```bash
pip install judgeval
```

Set your environment variables:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

Don't have keys? [Create an account](https://app.judgmentlabs.ai/register) on the platform!

## ✨ Judgeval Features in Detail

| Feature | Description | Benefits |
|---|---|---|
| **🧪 Evals** | Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure. | • Unit-testing • A/B testing • Online guardrails |
| **📡 Monitoring** | Get Slack alerts for agent failures in production. Add custom hooks to address production regressions. | • Identifying degradation early • Visualizing performance trends across agent versions and time |
| **📊 Datasets** | Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc. Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions. | • Agent environment interaction data for optimization • Scaled analysis for A/B tests |

## 🏢 Self-Hosting

Run Judgeval on your infrastructure for complete control:

**Key Benefits:**

*   Deploy on your own AWS account.
*   Store data in your Supabase instance.
*   Access via your custom domain.

**Getting Started:**

1.  Review the [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started).
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend.

## 📚 Cookbooks

Find and contribute to helpful Judgeval examples and best practices in our [cookbooks repository](https://github.com/JudgmentLabs/judgment-cookbook).  Contributions are welcome!

## 💻 Development with Cursor

Enhance your agent development experience with Cursor by utilizing the Judgeval rules file for better context and integration.  Find the rules file and documentation [here](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules).

## ⭐ Show Your Support!

If you like Judgeval, please give us a star on GitHub!  Your support helps the community grow and improves the project.

## ❤️ Contribute

We welcome contributions of all kinds:

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Improve the documentation through [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls).
*   Spread the word about Judgeval!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).