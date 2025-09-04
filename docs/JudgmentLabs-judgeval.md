<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Unleash the Power of Self-Learning Agents

**Judgeval empowers you to build and evaluate intelligent agents by providing crucial data and signals for continuous learning and self-improvement.** ([View on GitHub](https://github.com/JudgmentLabs/judgeval))

---

[Docs](https://docs.judgmentlabs.ai/) | [Judgment Cloud](https://app.judgmentlabs.ai/register) | [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) | [Landing Page](https://judgmentlabs.ai/)
[Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) | [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) | [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

---

## Key Features

*   **Evals:** Build robust evaluators to test and refine your agents using LLM-as-a-judge, manual labeling, and code-based evaluations.
    *   **Benefits:** Unit-testing, A/B testing, and online guardrails.
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>
*   **Monitoring:** Receive real-time alerts for agent failures and visualize performance trends to catch and address regressions early.
    *   **Benefits:** Identify degradation early, visualize performance trends across agent versions and time
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>
*   **Datasets:** Export and analyze agent-environment interactions to create datasets for in-depth optimization and scaling. Seamlessly integrate with Parquet, S3, and more.
    *   **Benefits:** Agent environment interaction data for optimization, Scaled analysis for A/B tests.
    <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

---

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

---

## 🛠️ Installation

Install the Judgeval SDK to start evaluating and monitoring your agents:

```bash
pip install judgeval
```

Configure your environment variables to connect to the Judgment Platform:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**[Create an account](https://app.judgmentlabs.ai/register) on the platform to obtain your API key and Organization ID.**

---

## 🏢 Self-Hosting

Gain full control over Judgeval by self-hosting on your infrastructure.

**Key Benefits:**

*   Deploy on your own AWS account.
*   Store data in your Supabase instance.
*   Access Judgeval through your custom domain.

**Getting Started:**

1.  Review the [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

---

## 📚 Cookbooks

Explore practical examples and recipes for using Judgeval in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook).  We welcome contributions!

---

## 💻 Development with Cursor

Enhance your coding experience in Cursor with Judgeval integration. The Cursor rules file, available in our documentation, provides the necessary context for effective agent and LLM workflow development.  Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules).

---

## ⭐ Contribute and Support

*   **Star us on GitHub:** If you find Judgeval valuable, please give us a star!
*   **Contribute:**
    *   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
    *   Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve it.

[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is proudly created and maintained by [Judgment Labs](https://judgmentlabs.ai/).