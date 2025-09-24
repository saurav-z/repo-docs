<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

<br>
<div style="font-size: 1.5em;">
    Enable self-learning agents with environment data and evals.
</div>

## [Docs](https://docs.judgmentlabs.ai/)  •  [Judgment Cloud](https://app.judgmentlabs.ai/register)  • [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)  • [Landing Page](https://judgmentlabs.ai/)

 [Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) • [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) • [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

</div>

# Judgeval: Build, Evaluate, and Optimize Autonomous Agents with Open-Source Tooling

**Judgeval empowers developers to build, monitor, and improve autonomous agents by providing a comprehensive open-source platform for evaluation and data-driven optimization.** ([See the original repository](https://github.com/JudgmentLabs/judgeval))

## Key Features

*   **Agent Evaluation:** Build custom evaluators using LLMs, manual labeling, or code to rigorously test and refine your agents.
*   **Real-time Monitoring:** Get instant Slack alerts for agent failures and visualize performance trends over time to quickly identify and address issues.
*   **Data-Driven Optimization:** Export agent-environment interactions into datasets for in-depth analysis, A/B testing, and continuous learning, leading to optimized performance.
*   **Self-Hosting:** Run Judgeval on your own infrastructure for full control over your data and environment.
*   **Integration with Cursor:** Enhance your development workflow with seamless integration with Cursor, a coding assistant designed to help you build and optimize agents.

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

## ✨ Key Features in Detail

### 🧪 Evals: Robust Evaluation Framework

Build custom evaluators on top of your agents using LLM-as-a-judge, manual labeling, or code-based approaches that integrate with our metric-tracking infrastructure.

**Useful for:**

*   ⚠️ Unit-testing
*   🔬 A/B testing
*   🛡️ Online guardrails

<p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

### 📡 Monitoring: Proactive Agent Management

Receive Slack alerts for agent failures in production and add custom hooks to address regressions immediately.

**Useful for:**

*   📉 Identifying degradation early
*   📈 Visualizing performance trends across agent versions and time

<p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

### 📊 Datasets: Data-Driven Optimization

Export environment interactions and test cases into datasets for scalable analysis and optimization. Move datasets to/from Parquet, S3, etc. Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions.

**Useful for:**

*   🗃️ Agent environment interaction data for optimization
*   🔄 Scaled analysis for A/B tests

<p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🏢 Self-Hosting: Own Your Infrastructure

Run Judgeval on your own infrastructure for complete control and customization.

### Key Features

*   Deploy Judgment on your own AWS account
*   Store data in your own Supabase instance
*   Access Judgment through your own custom domain

### Getting Started

1.  Check out our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  After setup, set the `JUDGMENT_API_URL` environmental variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore example implementations and use cases in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook) to accelerate your agent development. Contributions are welcome!  Create a PR or message us on [Discord](https://discord.gg/tGVFf8UBUY) to have your cookbook featured.

## 💻 Development with Cursor

Optimize your LLM workflows with Cursor and Judgeval integration. The Cursor rules file provides the context needed for effective use of Judgeval features within the coding assistant.

Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access to the rules file.

## ⭐ Star Us on GitHub

If you find Judgeval useful, please consider giving us a star on GitHub! Your support helps us grow our community and continue improving the repository.

## ❤️ Contributing

We welcome contributions from the community! Here are ways to get involved:

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
*   Review and improve the documentation through [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls)
*   Spread the word about Judgeval!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).