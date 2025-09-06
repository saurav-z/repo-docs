<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Build, Evaluate, and Optimize Self-Learning Agents

**Empower your self-learning agents with real-time environment data and robust evaluation tools.**

[**Explore Judgeval on GitHub**](https://github.com/JudgmentLabs/judgeval) | [Docs](https://docs.judgmentlabs.ai/) | [Judgment Cloud](https://app.judgmentlabs.ai/register) | [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) | [Landing Page](https://judgmentlabs.ai/)

[Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) | [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) | [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

Judgeval provides open-source tools for evaluating and improving autonomous agents, offering crucial runtime data for continuous learning and self-improvement.

## Key Features

*   **Comprehensive Agent Evaluation:** Define custom evaluators using LLMs, manual labeling, or code-based methods. Connect your evaluations with metric-tracking infrastructure.
*   **Proactive Agent Monitoring:** Receive real-time alerts via Slack for agent failures in production. Implement custom hooks to address production regressions.
*   **Data-Driven Optimization:** Export agent interactions and test cases into datasets for in-depth analysis and improvement. Easily move data to and from Parquet, S3, and more.

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

Get started with Judgeval by installing our SDK using pip:

```bash
pip install judgeval
```

Configure your environment variables to connect to the Judgment Platform:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have keys? [Create an account](https://app.judgmentlabs.ai/register) on the platform!**

## 🏢 Self-Hosting

Take full control with Judgeval's self-hosting capabilities. Run Judgment on your own infrastructure, with full control over the backend and data plane.

### Key Benefits:

*   Deploy Judgment on your own AWS account.
*   Store data in your own Supabase instance.
*   Access Judgment through your custom domain.

### Getting Started:

1.  Consult our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for comprehensive setup instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  Set the `JUDGMENT_API_URL` environment variable to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore practical examples and recipes for using Judgeval: [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook).  Contribute your own cookbook via a PR or message us on [Discord](https://discord.gg/tGVFf8UBUY).

## 💻 Development with Cursor

Improve your coding assistant's context with Judgeval integration using the Cursor rules file. Reference the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access and integration details.

## ⭐ Star Us on GitHub

Show your support by giving Judgeval a star on GitHub!

## ❤️ Contribute

We welcome contributions!  Here's how you can help:

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Review and improve our documentation via [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls).
*   Spread the word about Judgeval!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
```
Key improvements and SEO optimizations:

*   **Clear, Concise Title and Introduction:** The title is more descriptive, and the introduction directly communicates the core value proposition. The one-sentence hook is also present.
*   **Strategic Keyword Integration:**  Keywords like "self-learning agents," "evaluation tools," "monitoring," and "optimization" are naturally integrated throughout the content.
*   **Structured Content:** Uses headings, subheadings, and bullet points for readability and scannability, crucial for SEO.
*   **Emphasis on Benefits:** Highlights the advantages of using Judgeval (e.g., proactive monitoring, data-driven optimization) rather than just listing features.
*   **Strong Calls to Action:** Encourages users to explore the platform, check out the documentation, and create an account.
*   **Concise and Focused:**  Removes redundant phrases and prioritizes essential information.
*   **Enhanced Visual Appeal:** Includes relevant images and GIFs to break up text and illustrate key concepts.
*   **Consistent Formatting:** Uniform use of bolding, lists, and code blocks.
*   **Internal and External Linking:**  Optimized anchor text and links, directing users to relevant resources.
*   **Contributor Section:** Maintains original structure.

This revised README is more likely to attract and engage users, rank higher in search results, and effectively communicate the value of Judgeval.