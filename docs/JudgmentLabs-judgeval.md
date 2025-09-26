<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Empower Your Self-Learning Agents with Data and Evaluation

**Unlock the full potential of your autonomous agents with Judgeval, the open-source platform for monitoring, evaluating, and optimizing agent performance.** ([See the original repo](https://github.com/JudgmentLabs/judgeval))

## Key Features

*   **🚀 Real-time Monitoring:** Get instant alerts and visualize agent performance trends to catch regressions early.
*   **✅ Advanced Evaluation:** Build custom evaluators using LLMs, manual labeling, and code to unit-test and A/B test agent configurations.
*   **📊 Data-Driven Optimization:** Export agent interactions into datasets for scaled analysis, enabling continuous learning and environment-specific improvements.
*   **☁️ Cloud and Self-Hosting:** Choose the platform that best fits your needs with both cloud and self-hosting options.
*   **🛠️ Easy Integration:** Integrate Judgeval into your projects with a simple pip install and API key setup.

## 🎬 Judgeval in Action: Multi-Agent Research System

Judgeval provides complete observability for multi-agent systems. With just a few lines of code, you can capture environment responses across all agent tool calls.

**Watch the demo:**

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

Get started quickly by installing the Judgeval SDK:

```bash
pip install judgeval
```

Set your API keys:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

*   **Don't have API keys?** [Create a free account](https://app.judgmentlabs.ai/register) on the Judgment Platform.

## ✨ Judgeval Features in Detail

### 🧪 Evals
Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure.
*   ⚠️ Unit-testing
*   🔬 A/B testing
*   🛡️ Online guardrails

<p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

### 📡 Monitoring

Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.
*   📉 Identifying degradation early
*   📈 Visualizing performance trends across agent versions and time

<p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

### 📊 Datasets

Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc.
*   🗃️ Agent environment interaction data for optimization
*   🔄 Scaled analysis for A/B tests

<p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🏢 Self-Hosting

Take control of your data and infrastructure by self-hosting Judgeval.

**Key Benefits:**

*   Deploy on your own AWS account
*   Store data in your own Supabase instance
*   Access through your custom domain

**Get Started:**

1.  Follow our detailed [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started).
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore practical examples and integrations in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook) to get started.

## 💻 Development with Cursor

Enhance your development workflow by integrating Judgeval with Cursor, the AI-powered coding assistant. The [Cursor rules file](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) provides context to implement Judgeval features effectively.

## ⭐ Star Us on GitHub

Support the Judgeval community by giving us a star on GitHub!

## ❤️ Contributing

We welcome contributions! Check out ways you can contribute:

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
*   Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve it
*   Share your Judgeval experiences!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).