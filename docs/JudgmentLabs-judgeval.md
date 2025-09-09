<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Open-Source Tooling for Self-Learning Agents

**Supercharge your autonomous agents with real-time environment data and robust evaluation tools, enabling continuous learning and performance optimization.**

[**Get Started with Judgeval**](https://github.com/JudgmentLabs/judgeval)

## Key Features:

*   ✅ **Comprehensive Evaluation:** Build custom evaluators using LLMs, manual labeling, or code, connecting with our metric-tracking infrastructure.
*   📡 **Proactive Monitoring:** Receive Slack alerts for agent failures and integrate custom hooks to address regressions.
*   📊 **Data-Driven Datasets:** Export agent-environment interactions and test cases to datasets for scaled analysis and optimization.
*   🏢 **Self-Hosting:** Deploy Judgeval on your own infrastructure with full control over your data and backend.

---

## 🎬 See Judgeval in Action

**(1) Multi-Agent System with complete observability:** (1) A multi-agent system spawns agents to research topics on the internet. (2) With just **3 lines of code**, Judgeval captures all environment responses across all agent tool calls for monitoring. (3) After completion, (4) export all interaction data to enable further environment-specific learning and optimization.

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

---

## 🛠️ Installation

Install the Judgeval SDK using pip:

```bash
pip install judgeval
```

Set your environment variables to connect to the [Judgment Platform](https://app.judgmentlabs.ai/register):

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have an account? [Create one now](https://app.judgmentlabs.ai/register)!**

---

## ✨ Features in Depth

### 🧪 Evals
Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure.

**Useful for:**
*   ⚠️ Unit-testing
*   🔬 A/B testing
*   🛡️ Online guardrails

<p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

### 📡 Monitoring
Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.

**Useful for:**
*   📉 Identifying degradation early
*   📈 Visualizing performance trends across agent versions and time

<p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

### 📊 Datasets
Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc.

Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions.

**Useful for:**
*   🗃️ Agent environment interaction data for optimization
*   🔄 Scaled analysis for A/B tests

<p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

---

## 🏢 Self-Hosting

Take full control of your agent evaluation and data by self-hosting Judgeval.

### Key Advantages:

*   Deploy on your own AWS account
*   Store data in your own Supabase instance
*   Access through your custom domain

### Getting Started:

1.  Follow our detailed [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

---

## 📚 Cookbooks & Resources

*   **Cookbooks:** Explore practical examples and use cases in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook).
*   **Documentation:** [Comprehensive Documentation](https://docs.judgmentlabs.ai/) for Judgeval.

---

## 💻 Development with Cursor

Enhance your development workflow with Cursor by integrating Judgeval features. Refer to the [official documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access to the rules file and more information on integrating this rules file with your codebase.

---

## ⭐ Support & Community

*   **Star us on GitHub:** Show your support by starring the repository!
*   **Join the Discussion:** Connect with the community on [Discord](https://discord.gg/tGVFf8UBUY).
*   **Report Issues:**  Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   **Contribute:** Help us improve Judgeval by contributing through [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls).

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

**Made with ❤️ by [Judgment Labs](https://judgmentlabs.ai/).**