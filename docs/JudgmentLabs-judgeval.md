<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

<br>
<div style="font-size: 1.5em;">
    Empower your self-learning agents with real-time environment data and robust evaluation tools using Judgeval.
</div>

## [Docs](https://docs.judgmentlabs.ai/)  •  [Judgment Cloud](https://app.judgmentlabs.ai/register)  • [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)  • [Landing Page](https://judgmentlabs.ai/)

 [Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) • [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) • [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

</div>

## Judgeval: Open-Source Tooling for Autonomous Agent Evaluation

Judgeval is an open-source platform designed to provide crucial runtime data for evaluating, monitoring, and continuously improving autonomous, stateful agents.  **<a href="https://github.com/JudgmentLabs/judgeval">View the source code on GitHub.</a>**

## Key Features

*   **Evals:** Build custom evaluators using LLM-as-a-judge, manual labeling, or code-based methods. Connect with our metric-tracking infrastructure for comprehensive assessment.
    *   Useful for: Unit-testing, A/B testing, and Online guardrails.
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

*   **Monitoring:** Receive real-time Slack alerts for agent failures in production.  Implement custom hooks to address production regressions.
    *   Useful for: Identifying performance degradation early and Visualizing performance trends across agent versions and time.
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

*   **Datasets:** Export agent environment interactions and test cases into datasets.  Analyze, optimize, and A/B test different agent configurations using data exported to Parquet, S3, and more.
    *   Useful for: Agent environment interaction data for optimization and Scaled analysis for A/B tests.
    <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🎬 See Judgeval in Action

**[Multi-Agent System](https://github.com/JudgmentLabs/judgment-cookbook/tree/main/cookbooks/agents/multi-agent) with complete observability:**
1.  A multi-agent system spawns agents to research topics on the internet.
2.  With just **3 lines of code**, Judgeval captures all environment responses across all agent tool calls for monitoring.
3.  After completion,
4.  Export all interaction data to enable further environment-specific learning and optimization.

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

Take control of your agent evaluation infrastructure. Judgeval offers robust self-hosting capabilities, giving you full control over your backend and data plane.

### Key Benefits:

*   Deploy Judgeval on your own AWS account.
*   Store data in your own Supabase instance.
*   Access Judgeval through your own custom domain.

### Getting Started:

1.  Review our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  After setup, configure the `JUDGMENT_API_URL` environment variable to point to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore practical examples and recipes to get the most out of Judgeval. Find our cookbooks [here](https://github.com/JudgmentLabs/judgment-cookbook). We welcome your contributions! Create a PR or message us on [Discord](https://discord.gg/tGVFf8UBUY) to share your own.

## 💻 Development with Cursor

Optimize your coding workflow using Cursor!  The Cursor rules file provides key context for your coding assistant to implement Judgeval features effectively. Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access to the rules file and integration instructions.

## ⭐ Star Us on GitHub

Show your support and help us grow our community! Give Judgeval a star on GitHub.

## ❤️ Contributors

We appreciate contributions of all kinds!  Here's how you can get involved:

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
*   Review and improve our [documentation](https://github.com/JudgmentLabs/judgeval/pulls)
*   Spread the word and let us know what you're building!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).