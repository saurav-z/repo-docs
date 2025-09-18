<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Empowering Self-Learning Agents with Data and Evaluation

**Judgeval provides open-source tools to supercharge your autonomous agents by providing runtime data and evaluations for continuous learning and self-improvement.**

[GitHub Repository](https://github.com/JudgmentLabs/judgeval) | [Docs](https://docs.judgmentlabs.ai/) | [Judgment Cloud](https://app.judgmentlabs.ai/register) | [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) | [Landing Page](https://judgmentlabs.ai/)

[Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) | [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) | [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

## Key Features of Judgeval

*   **Evaluation:** Build custom evaluators using LLM-as-a-judge, manual labeling, or code-based metrics.  Integrate with our metric-tracking infrastructure for comprehensive assessment.
    *   Useful for: Unit-testing, A/B testing, and implementing online guardrails.
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>
*   **Monitoring:** Receive alerts for agent failures and identify performance degradation early. Visualize trends across agent versions and time.
    *   Useful for: Detecting regressions and visualizing performance trends.
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>
*   **Datasets:** Export agent-environment interactions to datasets for in-depth analysis and optimization. Seamlessly manage and move data to various formats (Parquet, S3, etc.)
    *   Useful for: Optimizing agent behavior and enabling scaled analysis for A/B testing.
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

Install the Judgeval SDK using pip:

```bash
pip install judgeval
```

Set your environment variables to connect to the [Judgment Platform](https://app.judgmentlabs.ai/):

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have API keys? [Create an account](https://app.judgmentlabs.ai/register) on the Judgment Platform!**

## 🏢 Self-Hosting

Run Judgeval on your infrastructure for complete control over your data.

### Key Benefits:

*   Deploy on your own AWS account.
*   Store data in your Supabase instance.
*   Access Judgeval through a custom domain.

### Getting Started:

1.  Follow the [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your instance.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Find example use cases and recipes in our cookbooks repository [here](https://github.com/JudgmentLabs/judgment-cookbook).  Create a PR or message us on [Discord](https://discord.gg/tGVFf8UBUY) to add your own!

## 💻 Development with Cursor

Enhance your coding experience with Cursor by integrating the Cursor rules file for efficient Judgment integration.  Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for setup instructions and the Cursor rules file.

## ⭐ Star Us on GitHub

If you find Judgeval useful, please consider giving us a star on GitHub!

## ❤️ Contribute

We welcome contributions!

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Improve documentation by submitting [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls).
*   Share your experiences with Judgment!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).