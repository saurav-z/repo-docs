<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Open-Source Tooling for Self-Learning Agents

**Unlock the power of self-learning agents with Judgeval, providing the data, evaluations, and monitoring tools you need.**

[**View the original repository on GitHub**](https://github.com/JudgmentLabs/judgeval)

*   [Docs](https://docs.judgmentlabs.ai/)
*   [Judgment Cloud](https://app.judgmentlabs.ai/register)
*   [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)
*   [Landing Page](https://judgmentlabs.ai/)
*   [Demo](https://www.youtube.com/watch?v=1S4LixpVbcc)
*   [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues)
*   [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

Judgeval is an open-source toolkit designed for evaluating and improving autonomous, stateful agents. It provides crucial runtime data from agent-environment interactions to fuel continuous learning and self-improvement.

## Key Features

*   **Evals:** Build custom evaluators using LLM-as-a-judge, manual labeling, or code-based evaluators, all integrated with our metric-tracking infrastructure.
    *   Useful for:
        *   Unit-testing
        *   A/B testing
        *   Online guardrails
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

*   **Monitoring:** Receive Slack alerts for agent failures in production and utilize custom hooks to address regressions.
    *   Useful for:
        *   Identifying degradation early
        *   Visualizing performance trends across agent versions and time
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

*   **Datasets:** Export agent environment interactions and test cases to datasets for in-depth analysis and optimization. Seamlessly move data to/from Parquet, S3, and more.
    *   Useful for:
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

Get started with Judgeval by installing our SDK using pip:

```bash
pip install judgeval
```

Ensure you have your `JUDGMENT_API_KEY` and `JUDGMENT_ORG_ID` environment variables set to connect to the [Judgment Platform](https://app.judgmentlabs.ai/).

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**If you don't have keys, [create an account](https://app.judgmentlabs.ai/register) on the platform!**

## 🏢 Self-Hosting

Take complete control by running Judgeval on your own infrastructure, with full access to the backend and data plane.

### Key Features
*   Deploy Judgment on your own AWS account
*   Store data in your own Supabase instance
*   Access Judgment through your own custom domain

### Getting Started
1.  Consult our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions.
2.  Utilize the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  After setup, ensure the `JUDGMENT_API_URL` environment variable points to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore practical examples and recipes in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook).  We welcome contributions; create a PR or reach out on [Discord](https://discord.gg/tGVFf8UBUY) to feature your own!

## 💻 Development with Cursor

Enhance your coding experience in Cursor by integrating the Judgment rules file for context-aware assistance with agent and LLM workflow development.  Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for the rules file and integration details.

## ⭐ Star Us on GitHub

Show your support and help us grow the community by giving Judgeval a star on GitHub!

## ❤️ Contributors

We appreciate contributions in many forms:

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
*   Improve the documentation via [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls)
*   Share your Judgeval experience!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is brought to you by [Judgment Labs](https://judgmentlabs.ai/).