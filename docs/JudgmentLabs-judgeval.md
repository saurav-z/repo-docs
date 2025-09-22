<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

## Judgeval: Supercharge Your Self-Learning Agents with Data and Evaluations

**Judgeval provides open-source tools to empower self-learning agents by capturing environment data and enabling robust evaluations.** 

[**View the original repository on GitHub**](https://github.com/JudgmentLabs/judgeval)

[Docs](https://docs.judgmentlabs.ai/) | [Judgment Cloud](https://app.judgmentlabs.ai/register) | [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) | [Landing Page](https://judgmentlabs.ai/) | [Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) | [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) | [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

Join us! We're hiring to build the future of self-learning agents.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

## Key Features

*   **Agent Evaluation:** Build custom evaluators using LLM-as-a-judge, manual labeling, or code-based approaches to monitor agent performance.
*   **Real-time Monitoring:** Get immediate alerts for agent failures and track performance trends, enabling rapid identification of issues.
*   **Dataset Creation:** Export environment interactions into datasets for in-depth analysis, A/B testing, and continuous agent improvement.
*   **Self-Hosting Capabilities:** Deploy Judgeval on your own infrastructure with full control over your data and backend.

## 🎬 See Judgeval in Action: Multi-Agent Research System

Observe a multi-agent system researching topics online. Judgeval captures all environment responses across agent tool calls to enable comprehensive monitoring and continuous learning.

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

*   [🛠️ Installation](#️-installation)
*   [✨ Features](#-features)
*   [🏢 Self-Hosting](#-self-hosting)
*   [📚 Cookbooks](#-cookbooks)
*   [💻 Development with Cursor](#-development-with-cursor)
*   [⭐ Star Us on GitHub](#-star-us-on-github)
*   [❤️ Contributors](#-contributors)

## 🛠️ Installation

Install the Judgeval SDK using pip:

```bash
pip install judgeval
```

Ensure the `JUDGMENT_API_KEY` and `JUDGMENT_ORG_ID` environment variables are configured. Create an account on the [Judgment Platform](https://app.judgmentlabs.ai/register) if you don't have keys.

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

## ✨ Features

| Feature | Description |
|---|---|
| **🧪 Evals** | Build custom evaluators for agent performance. Supports LLM-as-a-judge, manual labeling, and code-based evaluators.  Useful for: Unit-testing, A/B testing, and Online guardrails. <br><br> <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p> |
| **📡 Monitoring** | Get Slack alerts for agent failures. Add custom hooks to address regressions. Useful for: Identifying degradation early, and Visualizing performance trends. <br><br> <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p> |
| **📊 Datasets** | Export environment interactions and test cases for analysis and optimization. Run evals on datasets for unit tests or A/B tests. Useful for: Agent environment interaction data for optimization, and Scaled analysis for A/B tests. <br><br> <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p> |

## 🏢 Self-Hosting

Run Judgeval on your infrastructure:

*   Deploy on your own AWS account
*   Store data in your own Supabase instance
*   Access via your custom domain

### Getting Started

1.  Check our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started).
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation).
3.  Set `JUDGMENT_API_URL` to your self-hosted endpoint.

## 📚 Cookbooks

Contribute your own recipes by creating a PR or messaging us on [Discord](https://discord.gg/tGVFf8UBUY).

Access our cookbook repo [here](https://github.com/JudgmentLabs/judgment-cookbook).

## 💻 Development with Cursor

Integrate the Cursor rules file for improved context and efficient coding with your coding assistant: refer to our official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules).

## ⭐ Star Us on GitHub

Show your support by giving us a star on GitHub!

## ❤️ Contributors

We welcome contributions.

-   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
-   Review and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls)
-   Speak or write about Judgment.

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
```
Key improvements and SEO considerations:

*   **Clear, concise title:**  "Judgeval: Supercharge Your Self-Learning Agents" uses strong keywords.
*   **One-sentence hook:**  Immediately explains the core value proposition.
*   **Keywords:**  Repeatedly includes relevant keywords like "self-learning agents," "evaluations," "monitoring," and "datasets."
*   **Bulleted lists:** Uses bulleted lists to make features easy to scan and understand.
*   **Headings:**  Uses headings to structure the document, making it easy to navigate.
*   **Contextual links:** Provides links to related pages within the Judgment ecosystem.
*   **Clear Call to Action:** The "Star Us on GitHub" section encourages user engagement.
*   **Alt text for images:**  Includes descriptive alt text for all images, improving accessibility and SEO.
*   **Concise language:** The text is streamlined for readability.
*   **Self-hosting section:** Highlights a popular feature with a dedicated section.
*   **Contributors Section:** Encourages contributions.
*   **HTML Comments:** Removed unnecessary HTML comments.
*   **Removed unnecessary `div` tags:** For cleaner markdown formatting.
*   **Cleaned up table:** improved table formatting.