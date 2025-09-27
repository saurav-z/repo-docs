<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Empower Self-Learning Agents with Data & Evaluation

**Supercharge your autonomous agents with Judgeval, the open-source toolkit for runtime environment data, evaluations, and continuous improvement.**

[**View the Judgeval Repository on GitHub**](https://github.com/JudgmentLabs/judgeval)

## Key Features

*   **📊 Data-Driven Agent Optimization:** Capture and export agent-environment interactions to datasets for in-depth analysis, A/B testing, and continuous learning.  Move datasets to/from Parquet, S3, etc.
*   **🧪 Robust Evaluation Framework:** Build custom evaluators using LLMs, manual labeling, or code to validate agent performance, conduct unit tests, and establish online guardrails.
*   **📡 Real-Time Monitoring & Alerts:** Receive Slack alerts for agent failures and visualize performance trends across agent versions, ensuring optimal agent behavior.
*   **🔑 Self-Hosting Capabilities:** Deploy and manage Judgeval on your own infrastructure for full control over your data and backend.
*   **🛠️ Seamless Integration:** Utilize the Judgement CLI and Cursor Rules file for enhanced agent development and integration.

## 🎬 Judgeval in Action

See how Judgeval enables powerful multi-agent systems:

**(1) Multi-Agent Research:**  Spawn agents to research topics on the internet.
**(2) Data Capture:**  With just **3 lines of code**, capture all environment responses across all agent tool calls.
**(3) Completion & Analysis:**  After completion, export all interaction data for learning and optimization.

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

*   [🛠️ Installation](#-installation)
*   [✨ Features](#-features)
*   [🏢 Self-Hosting](#-self-hosting)
*   [📚 Cookbooks](#-cookbooks)
*   [💻 Development with Cursor](#-development-with-cursor)
*   [⭐ Contribute](#-contributors)

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

## ✨ Features

| Feature      | Description                                                                                                                                                                                                                                                                                  |
| :------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🧪 Evals** | Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure.                                                                                                             |
|               | **Useful for:**  <br>• ⚠️ Unit-testing <br>• 🔬 A/B testing <br>• 🛡️ Online guardrails                                                                                                                                                                                                |
| **📡 Monitoring** | Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.                                                                                                                                                                                           |
|               | **Useful for:** <br>• 📉 Identifying degradation early <br>• 📈 Visualizing performance trends across agent versions and time                                                                                                                                                                |
| **📊 Datasets** | Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc.                                                                                                                                                     |
|               | Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions.                                                                                                                                               |
|               | **Useful for:**<br>• 🗃️ Agent environment interaction data for optimization<br>• 🔄 Scaled analysis for A/B tests |

## 🏢 Self-Hosting

Run Judgeval on your own infrastructure for full control.

### Key Benefits:

*   Deploy on your own AWS account
*   Store data in your own Supabase instance
*   Access Judgeval through your custom domain

### Getting Started:

1.  Check out our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  After setup, set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore practical use cases and solutions in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook).  Contribute your own and get featured!

## 💻 Development with Cursor

Enhance your coding assistant's context for Judgment integration by utilizing the Cursor rules file. Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access and integration details.

## ⭐ Contribute

Help us improve Judgeval!

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Review and contribute to the [documentation](https://github.com/JudgmentLabs/judgeval/pulls).
*   Share your experiences and feedback on Judgment.

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).