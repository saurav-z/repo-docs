<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Empower Your Self-Learning Agents

**Supercharge your self-learning agents with real-time environment data and robust evaluation tools.**

[**Get Started with Judgeval!**](https://github.com/JudgmentLabs/judgeval)

## Key Features

*   **🤖 Agent Monitoring:** Get real-time insights and Slack alerts for agent failures, enabling proactive debugging and performance optimization.
*   **🧪 Automated Evals:** Build custom evaluators using LLMs, manual labeling, or code to rigorously test and refine your agents.
*   **📊 Data-Driven Optimization:** Export agent interactions to datasets for in-depth analysis, A/B testing, and continuous learning.
*   **📦 Self-Hosting:** Deploy Judgeval on your own infrastructure for complete control over your data and backend.

## What is Judgeval?

Judgeval is an open-source toolkit designed to provide the essential data and evaluation mechanisms needed for building and refining autonomous, stateful agents. It captures and analyzes agent-environment interactions, enabling developers to:

*   Monitor agent performance in real-time.
*   Automate agent evaluation and testing.
*   Optimize agents through data-driven insights.

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

## 🚀 Getting Started

1.  **Installation:** Install the Judgeval SDK:

    ```bash
    pip install judgeval
    ```

2.  **Configuration:** Set your API key and organization ID:

    ```bash
    export JUDGMENT_API_KEY=...
    export JUDGMENT_ORG_ID=...
    ```

    **Don't have API keys?** [Create an account](https://app.judgmentlabs.ai/register) on the Judgment Platform.

## 🛠️ Detailed Features

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

Run Judgeval on your own infrastructure for maximum control and data privacy.

### Key Benefits:

*   Deploy on your own AWS account.
*   Store data in your own Supabase instance.
*   Access Judgeval through your custom domain.

### Get Started:

1.  Refer to the [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  Set the `JUDGMENT_API_URL` environmental variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore practical examples and recipes in our [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook) to get you started!

## 💻 Development with Cursor

Enhance your LLM workflows in Cursor by integrating with Judgeval. Access the rules file for effective implementation of Judgment features:

Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access to the rules file and more information on integrating this rules file with your codebase.

## 🤝 Contribute

We welcome contributions! Here's how you can get involved:

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve it.
*   Share your experience with Judgeval on social media and let us know!

## ⭐ Star Us on GitHub

If you find Judgeval useful, please consider giving us a star on GitHub! Your support helps us grow our community and continue improving the repository.

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
```
Key improvements and summaries:

*   **SEO Optimization:** Added a clear, keyword-rich one-sentence hook and subheadings. The use of terms like "open-source," "agent monitoring," "evaluation," and "data-driven" increases searchability.
*   **Conciseness:** Removed redundant phrases and streamlined language.
*   **Emphasis on Benefits:** Focused on what users *gain* from using Judgeval.
*   **Clear Structure:** Improved readability with a well-defined table of contents (implicitly through headings) and bullet points.
*   **Call to Action:**  More prominent "Get Started" and "Star Us" calls to action.
*   **Updated Links:** Kept all original links intact and added a link back to the main repo at the top, making it easy for users to find the source code.
*   **Visual Appeal:** Maintained the image and GIF assets to make the README visually appealing and provide context.
*   **Clear Differentiation:**  Clearly distinguished the "Key Features" section from the more in-depth explanation.
*   **Improved Feature Descriptions:**  Enhanced descriptions of each feature to highlight benefits and use cases, improving comprehension and encouraging adoption.
*   **Simplified Installation:** Updated the installation steps and created a more user-friendly approach.