<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

<br>
<div style="font-size: 1.5em;">
    Unlock the power of self-learning agents with environment data and evaluations!
</div>

## Judgeval: Evaluate and Improve Your Autonomous Agents

**Judgeval** is an open-source toolkit that empowers you to evaluate, monitor, and continuously improve your autonomous, stateful agents.  Get started today and see how Judgeval can help you build better, more resilient AI agents.  [**Explore Judgeval on GitHub**](https://github.com/JudgmentLabs/judgeval)

**Key Features:**

*   **Evaluation:** Build custom evaluators using LLM-as-a-judge, manual labeling, or code-based solutions to test and refine your agents.
*   **Monitoring:** Receive real-time alerts and gain insights into agent performance with Slack integration, allowing you to quickly identify and address issues.
*   **Datasets:** Export agent interactions to datasets for scaled analysis, optimization, and A/B testing, enabling continuous learning from production interactions.
*   **Self-Hosting:** Deploy Judgeval on your infrastructure for complete control over your data and backend.

## 🎬 See Judgeval in Action

[Watch a demo](https://www.youtube.com/watch?v=1S4LixpVbcc) to see Judgeval in action, or dive directly into our [multi-agent system example](https://github.com/JudgmentLabs/judgment-cookbook/tree/main/cookbooks/agents/multi-agent).

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

## 🛠️ Installation

Get started with Judgeval by installing our SDK using pip:

```bash
pip install judgeval
```

Configure Judgeval by setting your `JUDGMENT_API_KEY` and `JUDGMENT_ORG_ID` environment variables to connect to the [Judgment Platform](https://app.judgmentlabs.ai/).

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have API keys? [Create an account](https://app.judgmentlabs.ai/register) on the Judgment Platform!**

## ✨ Features

| Feature        | Description                                                                                                                                                                                          | Use Cases                                                                                       |
| :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- |
| **🧪 Evals**     | Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure.                    | Unit-testing, A/B testing, Online guardrails                                                   |
| **📡 Monitoring** | Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.                                                                                               | Identifying degradation early, Visualizing performance trends across agent versions and time    |
| **📊 Datasets**   | Export environment interactions and test cases to datasets for scaled analysis and optimization.  Move datasets to/from Parquet, S3, etc.  Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions. | Agent environment interaction data for optimization, Scaled analysis for A/B tests |

<p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>
<p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>
<p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🏢 Self-Hosting

Take full control of your Judgeval setup with our self-hosting options.

### Key Benefits
*   Deploy Judgment on your own AWS account
*   Store data in your own Supabase instance
*   Access Judgment through your own custom domain

### Getting Started
1.  Consult our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  After setup, set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore example use cases and recipes in our [cookbooks](https://github.com/JudgmentLabs/judgment-cookbook) repository.  We welcome contributions!

## 💻 Development with Cursor

Enhance your coding experience within Cursor by integrating Judgeval's functionality. Refer to the [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for setting up your coding assistant with the necessary context to implement Judgment features effectively.

## ⭐ Star Us on GitHub

If you find Judgeval valuable, please consider giving us a star on GitHub! Your support helps us grow our community and continue improving the repository.

## ❤️ Contribute

We welcome contributions!  Help us improve Judgeval by:

*   Submitting [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Reviewing and improving the [documentation](https://docs.judgmentlabs.ai/) (and submitting [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls)).
*   Sharing your experiences with Judgeval.

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).