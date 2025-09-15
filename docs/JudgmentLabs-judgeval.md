<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

<br>
<div style="font-size: 1.5em;">
    Enable self-learning agents with environment data and evals.
</div>

## Empower Your AI Agents with Judgeval: Open-Source Evaluation and Monitoring

Judgeval provides powerful, open-source tools to build, monitor, and optimize autonomous agents, enabling continuous learning and improvement.  [Check out the original repo](https://github.com/JudgmentLabs/judgeval)!

[Docs](https://docs.judgmentlabs.ai/)  •  [Judgment Cloud](https://app.judgmentlabs.ai/register)  • [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)  • [Landing Page](https://judgmentlabs.ai/)

 [Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) • [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) • [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

</div>

## Key Features

Judgeval empowers you to:

*   **Evaluate Agent Performance:** Build custom evaluators using LLM-as-a-judge, manual labeling, and code-based metrics.
*   **Monitor Agent Health:** Receive Slack alerts for failures, identify performance degradation, and visualize trends.
*   **Analyze and Optimize with Data:** Export agent interactions to datasets for scaled analysis and A/B testing, enabling continuous learning.

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

Install Judgeval with pip:

```bash
pip install judgeval
```

Set your API keys to connect to the [Judgment Platform](https://app.judgmentlabs.ai/).

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**[Create an account](https://app.judgmentlabs.ai/register) if you don't have one!**

## ✨ Feature Breakdown

*   **Evals:**  Build custom evaluators including LLM-as-a-judge, manual labeling, and code-based metrics, and connect with metric-tracking infrastructure.
    *   **Use Cases:** Unit-testing, A/B testing, and online guardrails.
    *   <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>
*   **Monitoring:** Get Slack alerts for agent failures, and add hooks to address regressions.
    *   **Use Cases:**  Identifying degradation early and visualizing performance trends.
    *   <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>
*   **Datasets:** Export agent interactions and test cases to datasets for analysis and optimization, supporting various formats.
    *   **Use Cases:** Agent environment interaction data for optimization and scaled A/B tests.
    *   <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## 🏢 Self-Hosting

Take full control of your Judgeval instance with self-hosting.

### Key Benefits:

*   Deploy on your own AWS account.
*   Store data in your own Supabase instance.
*   Access through your custom domain.

### Getting Started

1.  Follow the [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started).
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy.
3.  Set `JUDGMENT_API_URL` to your self-hosted backend endpoint.

## 📚 Cookbooks

Explore the [Judgment Cookbook](https://github.com/JudgmentLabs/judgment-cookbook) for example use cases.  Contribute your own by creating a PR or contacting us on [Discord](https://discord.gg/tGVFf8UBUY).

## 💻 Development with Cursor

Integrate Judgeval with Cursor for enhanced coding assistance. Refer to the [official documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access to the rules file.

## ⭐ Star Us on GitHub!

Show your support by giving Judgeval a star on GitHub!

## ❤️ Contributing

Contribute to Judgeval by:

*   Submitting [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues).
*   Reviewing the documentation and submitting [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls).
*   Sharing your experiences with Judgeval!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
```

Key improvements and SEO optimizations:

*   **Clear Headline:** A strong, SEO-friendly headline.
*   **Concise Summary:** A one-sentence hook to grab attention and summarize the project.
*   **Keyword Rich:**  Uses relevant keywords like "AI agents", "evaluation", "monitoring", and "open-source" throughout.
*   **Structured Content:** Uses headings, bullet points, and tables to improve readability and SEO ranking.
*   **Actionable Language:** Uses verbs to encourage engagement ("Empower", "Build", "Monitor", "Optimize").
*   **Focus on Benefits:** Highlights the *value* Judgeval provides.
*   **Internal Linking:**  Links to key resources within the README and the original repo.
*   **Call to Action:** Encourages users to star the repo.
*   **Alt Text for Images:**  Provides alt text for all images, improving accessibility and SEO.
*   **Keyword Density:** Ensures important keywords are used naturally throughout the text.
*   **Concise & Clear Language:** Rewrites sections for better flow.
*   **Includes a "Getting Started" Section:**  Addresses the needs of new users directly.
*   **Updated Social Links**  Includes X/Twitter, LinkedIn and Discord links.