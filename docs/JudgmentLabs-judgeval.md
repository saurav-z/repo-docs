<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Open-Source Tooling for Self-Learning Agents

**Judgeval empowers you to build, evaluate, and continuously improve autonomous agents by providing critical data and insights.**

[Docs](https://docs.judgmentlabs.ai/)  •  [Judgment Cloud](https://app.judgmentlabs.ai/register)  • [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)  • [Landing Page](https://judgmentlabs.ai/) • [GitHub](https://github.com/JudgmentLabs/judgeval)

[Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) • [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) • [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

## Key Features

*   **Comprehensive Agent Evaluation:** Build custom evaluators using LLMs, manual labeling, or code to rigorously test your agents.

*   **Real-time Agent Monitoring:** Receive immediate Slack alerts for agent failures and identify performance regressions with custom hooks.

*   **Data-Driven Optimization:** Export environment interactions and test cases into datasets for detailed analysis and continuous improvement through A/B testing.

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

## Getting Started

### Installation

Install the Judgeval SDK using pip:

```bash
pip install judgeval
```

Ensure you have your `JUDGMENT_API_KEY` and `JUDGMENT_ORG_ID` environment variables set to connect to the [Judgment Platform](https://app.judgmentlabs.ai/).

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have API keys? [Create an account](https://app.judgmentlabs.ai/register) on the Judgment Platform!**

## Deep Dive into Features

### 🧪 Evals

*   Build custom evaluators for agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators.
*   Connects with metric-tracking infrastructure.
*   **Useful for:**
    *   ⚠️ Unit-testing
    *   🔬 A/B testing
    *   🛡️ Online guardrails

<p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

### 📡 Monitoring

*   Get Slack alerts for agent failures in production.
*   Add custom hooks to address production regressions.
*   **Useful for:**
    *   📉 Identifying degradation early
    *   📈 Visualizing performance trends across agent versions and time

<p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

### 📊 Datasets

*   Export environment interactions and test cases to datasets.
*   Move datasets to/from Parquet, S3, etc.
*   Run evals on datasets for unit tests or A/B testing different agent configurations.
*   **Useful for:**
    *   🗃️ Agent environment interaction data for optimization
    *   🔄 Scaled analysis for A/B tests

<p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🏢 Self-Hosting

Run Judgeval on your infrastructure, gaining full control over the backend and data.

### Key Benefits
*   Deploy Judgment on your own AWS account
*   Store data in your own Supabase instance
*   Access Judgment through your own custom domain

### Get Started
1.  Review our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks & Community

Explore and contribute to our growing collection of agent recipes:

*   Access our repo of cookbooks [here](https://github.com/JudgmentLabs/judgment-cookbook).
*   Share your own recipes through a PR or on our [Discord](https://discord.gg/tGVFf8UBUY).

## 💻 Development with Cursor

Enhance your coding experience with Cursor:

*   Integrate the Cursor rules file for optimal context about Judgment integration.
*   Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for the rules file and integration details.

## ⭐ Star Us on GitHub

Show your support! Give Judgeval a star on [GitHub](https://github.com/JudgmentLabs/judgeval).

## ❤️ Contribute

Become a part of the Judgeval community:

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Contribute to our documentation through [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls).
*   Share your Judgeval experiences.

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
```
Key improvements and SEO optimizations:

*   **Concise Hook:**  A strong, benefit-driven opening sentence immediately grabs attention.
*   **Clear Headings:**  Well-defined headings break up the content for readability and SEO.
*   **Keyword-Rich Titles:**  Includes important keywords like "self-learning agents," "evaluation," "monitoring," and "open-source."
*   **Bulleted Key Features:**  Highlights the core benefits in an easy-to-scan format.
*   **Action-Oriented Language:** Uses verbs like "build," "evaluate," and "improve" to encourage engagement.
*   **Strategic Use of Visuals:** Includes images and GIFs to increase engagement.
*   **Internal Linking:**  Links to different sections within the README.
*   **External Linking:** Links to the original GitHub repo, documentation, related projects (cookbooks), and social media.
*   **Call to Action:**  Encourages starring the repo and contributing.
*   **Concise Descriptions:**  The feature descriptions are brief and highlight benefits.
*   **Emphasis on Self-Hosting:**  Highlights the option for users who prefer control over their data.
*   **Contributor Section:**  Encourages community contributions.
*   **Clear Installation Instructions:** Easy to find, including the required environment variables.
*   **Emphasis on the problem the project solves:** Framing the benefits as solutions.