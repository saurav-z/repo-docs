<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

<br>
</div>

# Judgeval: Empowering Self-Learning Agents with Data and Evaluations

**Unlock the power of self-learning agents with Judgeval, providing the crucial data and evaluation tools needed for continuous improvement. Check out the original repo [here](https://github.com/JudgmentLabs/judgeval).**

## Key Features

*   **🚀 Comprehensive Evaluation Framework:** Build custom evaluators using LLMs, manual labeling, or code, connecting with robust metric-tracking infrastructure.
*   **👁️ Real-time Monitoring:** Receive Slack alerts for agent failures and integrate custom hooks to address production regressions, allowing for early identification and resolution of performance drops.
*   **📊 Data-Driven Datasets:** Export agent interactions to datasets for in-depth analysis and optimization. Integrate with Parquet, S3 and other formats to run evals for A/B testing.

## 🎬 See Judgeval in Action

Watch Judgeval in action with this [Multi-Agent System Demo](https://github.com/JudgmentLabs/judgment-cookbook/tree/main/cookbooks/agents/multi-agent). Observe agents research topics with complete observability:

1.  A multi-agent system spawns agents.
2.  Capture environment responses with just **3 lines of code**.
3.  Export interaction data for environment-specific learning.

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

Get started with Judgeval:

```bash
pip install judgeval
```

Ensure your `JUDGMENT_API_KEY` and `JUDGMENT_ORG_ID` environment variables are set to connect to the [Judgment Platform](https://app.judgmentlabs.ai/).

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have keys? [Create an account](https://app.judgmentlabs.ai/register) on the platform!**

## ✨ Features in Detail

### 🧪 Evals

*   **Description:** Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure.
*   **Use Cases:** Unit-testing, A/B testing, and Online guardrails.
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

### 📡 Monitoring

*   **Description:** Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.
*   **Use Cases:** Identifying early degradation, and visualizing performance trends across agent versions and time.
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

### 📊 Datasets

*   **Description:** Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc.
*   **Use Cases:** Agent environment interaction data for optimization, and scaled analysis for A/B tests.
    <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🏢 Self-Hosting

Take control with Judgeval's self-hosting capabilities.

### Key Benefits:

*   Deploy on your own AWS account
*   Store data in your own Supabase instance
*   Access Judgeval through your own custom domain

### Getting Started

1.  Follow the detailed setup instructions in our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started).
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore recipes and best practices in our [cookbook repository](https://github.com/JudgmentLabs/judgment-cookbook).

## 💻 Development with Cursor

Enhance your coding experience with Cursor by integrating the Judgment rules file from the [official documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules).

## ⭐ Star Us on GitHub

Show your support by starring Judgeval on [GitHub](https://github.com/JudgmentLabs/judgeval)!

## ❤️ Contributors

Contribute to Judgeval!

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
*   Submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls)
*   Spread the word about Judgment!

[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is a project by [Judgment Labs](https://judgmentlabs.ai/).