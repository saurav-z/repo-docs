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

## Judgeval: Unlock the Power of Self-Learning Agents with Data and Evaluation

**Judgeval provides open-source tools for evaluating, monitoring, and optimizing autonomous agents, empowering them to learn and improve continuously.**

[**Visit the Judgeval GitHub Repository**](https://github.com/JudgmentLabs/judgeval)

### Key Features

*   **Evaluation:** Build custom evaluators using LLMs, manual labeling, or code-based solutions to assess agent performance.
*   **Monitoring:** Receive real-time alerts for agent failures and track performance trends to identify and address issues quickly.
*   **Datasets:** Export agent interactions into datasets for scaled analysis and optimization, enabling A/B testing and continuous learning.
*   **Self-Hosting:** Deploy Judgeval on your own infrastructure for complete control over your data and platform.

### 🎬 See Judgeval in Action

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

Easily install the Judgeval SDK using pip:

```bash
pip install judgeval
```

Configure your environment variables to connect to the [Judgment Platform](https://app.judgmentlabs.ai/).

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have API keys? [Create an account](https://app.judgmentlabs.ai/register)!**

## ✨ Features

*   ### 🧪 Evals
    Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure.
    <br><br>**Useful for:**<br>• ⚠️ Unit-testing <br>• 🔬 A/B testing <br>• 🛡️ Online guardrails
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

*   ### 📡 Monitoring
    Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.
    <br><br> **Useful for:** <br>• 📉 Identifying degradation early <br>• 📈 Visualizing performance trends across agent versions and time
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

*   ### 📊 Datasets
    Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc.
    <br><br>Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions.
    <br><br> **Useful for:**<br>• 🗃️ Agent environment interaction data for optimization<br>• 🔄 Scaled analysis for A/B tests
    <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🏢 Self-Hosting

Take control with Judgeval by self-hosting on your infrastructure.

### Key Benefits

*   Deploy on your AWS account
*   Store data in your Supabase instance
*   Access via your custom domain

### Getting Started

1.  Consult the [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore our collection of cookbooks [here](https://github.com/JudgmentLabs/judgment-cookbook) or contribute your own via PR or Discord.

## 💻 Development with Cursor

Integrate Judgeval seamlessly into your Cursor development environment. Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) to access the rules file for implementing Judgeval features effectively.

## ⭐ Star Us on GitHub

Show your support by starring Judgeval on GitHub!

## ❤️ Contributors

Contribute to Judgeval by submitting bug reports, feature requests, documentation improvements, or sharing your experiences.

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
```
Key improvements and explanations:

*   **SEO Optimization:**  Used keywords like "self-learning agents," "evaluation," "monitoring," and "optimization" in the headings and text.  Added a concise, benefit-driven hook.  Focused on what Judgeval *does* for the user, not just what it is.
*   **Clear Structure:**  Used clear headings and subheadings for easy navigation.  Emphasized the benefits of each feature with bullet points and visual aids (images already in the original).
*   **Concise Language:**  Trimmed down lengthy sentences and focused on the core value proposition.
*   **Action-Oriented:**  Added calls to action (e.g., "Visit the Judgeval GitHub Repository," "Create an account") to guide the user.
*   **Benefit-Driven:**  Framed features in terms of their user benefits ("Unlock the Power...", "Empowering them to learn...").
*   **Readability:** Improved the formatting for better readability, especially with the feature descriptions.
*   **Removed Redundancy:**  Streamlined the "Installation" section.
*   **Clearer Explanations:**  Made the "Self-Hosting" and "Cursor Development" sections more understandable.
*   **Markdown Formatting:**  Ensured the markdown is clean and consistent.
*   **GitHub Link:** Explicitly included a link back to the GitHub repository at the beginning.
*   **Summarized the "Features" section and added more context.**
*   **More impactful intro before installation section**

This revised README is more engaging, easier to understand, and more likely to attract users and contributors.  It's also optimized for search engines to improve visibility.