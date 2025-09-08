<div align="center">

<img src="assets/new_lightmode.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/new_darkmode.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

</div>

# Judgeval: Empower Your Self-Learning Agents with Data and Evaluations

**Judgeval provides open-source tooling to supercharge autonomous agents by capturing environment data and enabling robust evaluation, leading to continuous learning and improvement.**

[**GitHub Repository**](https://github.com/JudgmentLabs/judgeval) | [Docs](https://docs.judgmentlabs.ai/) | [Judgment Cloud](https://app.judgmentlabs.ai/register) | [Self-Host](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) | [Landing Page](https://judgmentlabs.ai/)

[Demo](https://www.youtube.com/watch?v=1S4LixpVbcc) | [Bug Reports](https://github.com/JudgmentLabs/judgeval/issues) | [Changelog](https://docs.judgmentlabs.ai/changelog/2025-04-21)

We're hiring! Join us in our mission to enable self-learning agents by providing the data and signals needed for monitoring and post-training.

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />

## Key Features

*   **Evaluation:** Build custom evaluators using LLM-as-a-judge, manual labeling, or code-based methods, integrated with our metric-tracking infrastructure.
    *   **Benefits:** Unit-testing, A/B testing, and online guardrails.
    <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p>

*   **Monitoring:** Receive Slack alerts for agent failures and utilize custom hooks to address regressions.
    *   **Benefits:** Early detection of degradation and visualization of performance trends.
    <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p>

*   **Datasets:** Export agent interactions and test cases to datasets for analysis and optimization. Easily move data to/from Parquet, S3, etc.
    *   **Benefits:**  Agent environment data for optimization and scalable analysis.
    <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p>

## Judgeval in Action

Observe how Judgeval enhances a multi-agent system:

**[Multi-Agent System](https://github.com/JudgmentLabs/judgment-cookbook/tree/main/cookbooks/agents/multi-agent) with complete observability:**

| Agents Running                                                                                                    | Capturing Environment Data                                                                                             | Agents Completed Running                                                                                                 | Exporting Agent Environment Data                                                                                              |
| :---------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------- |
| <img src="assets/agent.gif" alt="Agent Demo" style="width: 100%; max-width: 350px; height: auto;" />           | <img src="assets/trace.gif" alt="Capturing Environment Data Demo" style="width: 100%; max-width: 350px; height: auto;" /> | <img src="assets/document.gif" alt="Agent Completed Demo" style="width: 100%; max-width: 350px; height: auto;" />      | <img src="assets/data.gif" alt="Data Export Demo" style="width: 100%; max-width: 350px; height: auto;" />               |

## Installation

Install the Judgeval SDK using pip:

```bash
pip install judgeval
```

Set your API key and organization ID:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**Don't have an account? [Create one](https://app.judgmentlabs.ai/register) on the Judgment Platform!**

## Self-Hosting

Take control of your data and infrastructure: Judgeval offers comprehensive self-hosting capabilities.

### Key Benefits:
*   Deploy Judgment on your own AWS account
*   Store data in your Supabase instance
*   Access Judgment through your custom domain

### Getting Started:

1.  Refer to our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for instructions.
2.  Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment.
3.  Ensure the `JUDGMENT_API_URL` environment variable points to your self-hosted backend.

## Cookbooks

Explore example use cases and recipes in our [cookbooks](https://github.com/JudgmentLabs/judgment-cookbook). We welcome your contributions!

## Development with Cursor

Enhance your coding workflow with Cursor by integrating Judgeval's features.  Access the rules file and integrate it into your codebase for effective development.
Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for more information.

## Contribute

We welcome contributions!

*   Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
*   Suggest [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve the documentation.
*   Share your work with the community!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).