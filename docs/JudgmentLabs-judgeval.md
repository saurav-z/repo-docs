<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

<br>
</div>

# Judgeval: Supercharge Your Self-Learning Agents

**Easily evaluate and optimize your autonomous agents with environment data and powerful evaluation tools.** ([View the original repository](https://github.com/JudgmentLabs/judgeval))

## Key Features

*   **Comprehensive Agent Evaluation:** Build custom evaluators using LLM-as-a-judge, manual labeling, or code-based methods.
*   **Real-Time Monitoring:** Receive Slack alerts for agent failures and visualize performance trends.
*   **Data-Driven Optimization:** Export and analyze environment interactions to create datasets for scaled analysis and A/B testing.
*   **Self-Hosting Options:** Deploy Judgeval on your own infrastructure for full control over your data.
*   **Seamless Integration:** Easily integrates with popular tools and environments.

## Table of Contents

*   [🎬 See Judgeval in Action](#see-judgeval-in-action)
*   [🛠️ Installation](#installation)
*   [✨ Features](#features)
*   [🏢 Self-Hosting](#self-hosting)
*   [📚 Cookbooks](#cookbooks)
*   [💻 Development with Cursor](#development-with-cursor)
*   [⭐ Star Us on GitHub](#-star-us-on-github)
*   [❤️ Contributors](#-contributors)

## 🎬 See Judgeval in Action

Judgeval provides open-source tooling for evaluating autonomous, stateful agents. It provides runtime data from agent-environment interactions for continuous learning and self-improvement.

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

## ✨ Features

| Feature          | Description                                                                                                                                                                                                                |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🧪 Evals**       | Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure. <br><br>**Useful for:**<br>• ⚠️ Unit-testing <br>• 🔬 A/B testing <br>• 🛡️ Online guardrails |
| **📡 Monitoring**  | Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.<br><br> **Useful for:** <br>• 📉 Identifying degradation early <br>• 📈 Visualizing performance trends across agent versions and time        |
| **📊 Datasets**   | Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc. <br><br>Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions. <br><br> **Useful for:**<br>• 🗃️ Agent environment interaction data for optimization<br>• 🔄 Scaled analysis for A/B tests                |
|                   |                                                                                                                                                                                    |

## 🏢 Self-Hosting

Run Judgment on your own infrastructure: we provide comprehensive self-hosting capabilities that give you full control over the backend and data plane that Judgeval interfaces with.

### Key Features

*   Deploy Judgment on your own AWS account
*   Store data in your own Supabase instance
*   Access Judgment through your own custom domain

### Getting Started

1.  Check out our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions, along with how your self-hosted instance can be accessed
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment
3.  After your self-hosted instance is setup, make sure the `JUDGMENT_API_URL` environmental variable is set to your self-hosted backend endpoint

## 📚 Cookbooks

Find pre-built, practical solutions to common agent-related problems in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook).

We welcome your contributions!  Create a PR or message us on [Discord](https://discord.gg/tGVFf8UBUY) to feature your own cookbook.

## 💻 Development with Cursor

Optimize your LLM workflows and agent development by integrating Judgeval with Cursor. The Cursor rules file contains key information to effectively implement Judgeval features within Cursor.

Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access to the rules file and more information on integrating this rules file with your codebase.

## ⭐ Star Us on GitHub

If you find Judgeval useful, please consider giving us a star on GitHub! Your support helps us grow our community and continue improving the repository.

## ❤️ Contributors

There are many ways to contribute to Judgeval:

-   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
-   Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve it
-   Speaking or writing about Judgment and letting us know!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).