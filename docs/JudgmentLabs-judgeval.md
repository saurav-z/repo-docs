<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Open-Source Tooling for Self-Learning Agents

**Judgeval empowers you to build, monitor, and continuously improve autonomous agents by providing runtime data and evaluation tools.**  [View the original repository](https://github.com/JudgmentLabs/judgeval).

**Key Features:**

*   ✅ **Real-time Evaluation:** Build custom evaluators (LLM-as-a-judge, manual labeling, code-based) to assess agent performance, perform A/B testing and implement online guardrails.
*   📡 **Comprehensive Monitoring:** Receive alerts for agent failures and visualize performance trends with real-time dashboards and Slack integrations.
*   📊 **Data-Driven Optimization:** Export agent interactions to datasets for detailed analysis, A/B testing, and continuous learning.

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

Install the Judgeval SDK with pip:

```bash
pip install judgeval
```

Set your environment variables:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

Need keys? [Create an account](https://app.judgmentlabs.ai/register) on the platform.

## ✨ Features Deep Dive

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

## 🏢 Self-Hosting

Run Judgeval on your infrastructure for complete control.

**Key Benefits:**

*   Deploy on your AWS account.
*   Store data in your Supabase instance.
*   Access through your custom domain.

**Get Started:**

1.  Follow the [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started).
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy.
3.  Set `JUDGMENT_API_URL` to your self-hosted endpoint.

## 📚 Cookbooks

Explore example implementations in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook).  Contributions are welcome!

## 💻 Development with Cursor

Improve your Cursor experience by integrating Judgeval context.  Refer to the [official documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for the rules file.

## ⭐ Star Us on GitHub

Show your support!  Give us a star on GitHub to help grow the community.

## ❤️ Contribute

Help improve Judgeval!  Contribute by:

*   Submitting [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Improving the documentation with [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls).
*   Spreading the word!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).