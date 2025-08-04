<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: Supercharge Your AI Engineering with Observability, Evaluations, and More!

[OpenLIT](https://github.com/openlit/openlit) simplifies your AI development workflow, providing essential tools for building and deploying robust AI applications.

**Key Features:**

*   📈 **Comprehensive Observability:** Monitor your AI application's health and performance.
*   🔌 **OpenTelemetry-native SDKs:** Integrate seamlessly with your existing observability tools.
*   💲 **Cost Tracking:** Estimate model costs precisely.
*   🐛 **Exception Monitoring:** Quickly identify and resolve errors.
*   💭 **Prompt Management:** Organize and version prompts effectively.
*   🔑 **Secure Secrets Management:** Protect API keys and secrets.
*   🎮 **LLM Experimentation:** Test and compare different LLMs.
*   🚀 **Full-Stack Monitoring:** Observe LLMs, vector databases, and GPUs with a single line of code.

**[Documentation](https://docs.openlit.io/) | [Quickstart](-getting-started-with-llm-observability) | [Python SDK](https://github.com/openlit/openlit/tree/main/sdk/python) | [Typescript SDK](https://github.com/openlit/openlit/tree/main/sdk/typescript)**

**[Roadmap](#️-roadmap) | [Feature Request](https://github.com/openlit/openlit/issues/new?assignees=&labels=%3Araised_hand%3A+Up+for+Grabs%2C+%3Arocket%3A+Feature&projects=&template=feature-request.md&title=%5BFeat%5D%3A) | [Report a Bug](https://github.com/openlit/openlit/issues/new?assignees=&labels=%3Abug%3A+Bug%2C+%3Araised_hand%3A+Up+for+Grabs&projects=&template=bug.md&title=%5BBug%5D%3A)**

[![OpenLIT](https://img.shields.io/badge/OpenLIT-orange)](https://openlit.io/)
[![License](https://img.shields.io/github/license/openlit/openlit?label=License&logo=github&color=f80&logoColor=white)](https://github.com/openlit/openlit/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/openlit/month)](https://pepy.tech/project/openlit)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/openlit/openlit)](https://github.com/openlit/openlit/pulse)
[![GitHub Contributors](https://img.shields.io/github/contributors/openlit/openlit)](https://github.com/openlit/openlit/graphs/contributors)

[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ)
[![X](https://img.shields.io/badge/follow-%40openlit__io-1DA1F2?logo=x&style=social)](https://twitter.com/openlit_io)

---

## Getting Started

### Step 1: Deploy OpenLIT Stack

1.  **Clone the Repository:**

    ```shell
    git clone git@github.com:openlit/openlit.git
    ```

2.  **Self-host with Docker:**

    ```shell
    docker compose up -d
    ```

    > For Kubernetes installation instructions, see the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes).

### Step 2: Install OpenLIT SDK

```bash
pip install openlit
```

> For the TypeScript SDK, visit the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).

### Step 3: Initialize in Your Application

Add the following lines to your application code:

```python
import openlit
openlit.init()
```

Configure your telemetry destination:

| Purpose                            | Parameter/Environment Variable                   | Example                               |
| ---------------------------------- | ------------------------------------------------ | ------------------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"`             |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | (Not required by default)             |

> 💡 **Tip:**  If `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` is not provided, traces will output to the console.

---

<details>
  <summary>Initialize using Function Arguments</summary>

  ```python
  import openlit

  openlit.init(
    otlp_endpoint="http://127.0.0.1:4318",
  )
  ```
</details>

---
<details>
  <summary>Initialize using Environment Variables</summary>

  ```python
  import openlit

  openlit.init()
  ```

  Then, configure your OTLP endpoint:

  ```env
  export OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318"
  ```
</details>
---

### Step 4: Visualize and Optimize

Access your data at `127.0.0.1:3000` to begin exploring the insights. Login using:

-   **Email:** `user@openlit.io`
-   **Password:** `openlituser`

## ⚡ Features

![OpenLIT Banner](https://github.com/openlit/.github/blob/main/profile/assets/openlit-feature-banner.png?raw=true)

-   📈 **Analytics Dashboard:** Monitor your AI application's health and performance with detailed dashboards.
-   🔌 **OpenTelemetry-native Observability SDKs:** Vendor-neutral SDKs to send traces and metrics.
-   💲 **Cost Tracking:** Tailor cost estimations for specific models.
-   🐛 **Exceptions Monitoring Dashboard:** Quickly spot and resolve issues.
-   💭 **Prompt Management:** Manage and version prompts using Prompt Hub.
-   🔑 **API Keys and Secrets Management:** Securely handle your API keys and secrets centrally.
-   🎮 **Experiment with different LLMs:** Use OpenGround to explore, test and compare various LLMs side by side.

## 🛣️ Roadmap

Here's a look at recent accomplishments and upcoming features:

| Feature                                                                                                                           | Status        |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| [OpenTelemetry-native Observability SDK for Tracing and Metrics](https://github.com/openlit/openlit/tree/text-upgrade/sdk/python) | ✅ Completed   |
| [OpenTelemetry-native GPU Monitoring](https://docs.openlit.io/latest/features/gpu)                                                | ✅ Completed   |
| [Exceptions and Error Monitoring](https://docs.openlit.io/latest/features/exceptions)                                             | ✅ Completed   |
| [Prompt Hub for Managing and Versioning Prompts](https://docs.openlit.io/latest/features/prompt-hub)                              | ✅ Completed   |
| [OpenGround for Testing and Comparing LLMs](https://docs.openlit.io/latest/features/openground)                                   | ✅ Completed   |
| [Vault for Central Management of LLM API Keys and Secrets](https://docs.openlit.io/latest/features/vault)                         | ✅ Completed   |
| [Cost Tracking for Custom Models](https://docs.openlit.io/latest/features/pricing)                                                | ✅ Completed   |
| [Real-Time Guardrails Implementation](https://docs.openlit.io/latest/features/guardrails)                                         | ✅ Completed   |
| [Programmatic Evaluation for LLM Response](https://docs.openlit.io/latest/features/evaluations)                                   | ✅ Completed   |
| [Auto-Evaluation Metrics Based on Usage](https://github.com/openlit/openlit/issues/470)                                           | 🔜 Coming Soon |
| [Human Feedback for LLM Events](https://github.com/openlit/openlit/issues/471)                                                    | 🔜 Coming Soon |
| [Dataset Generation Based on LLM Events](https://github.com/openlit/openlit/issues/472)                                           | 🔜 Coming Soon |
| [Search over Traces]()                                                                                                            | 🔜 Coming Soon |

## 🌱 Contributing

Contributions are welcome! See our [Contribution guide](./CONTRIBUTING.md).

Get involved by:

-   Joining our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) community.

[![OpenLIT - One click observability, evals for LLMs & GPUs | Product Hunt](https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=460690&theme=light)](https://www.producthunt.com/posts/openlit?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-openlit)
<a href="https://fazier.com/launches/openlit-2" target="_blank" rel="noopener noreferrer"><img src="https://fazier.com/api/v1/public/badges/embed_image.svg?launch_id=779&badge_type=daily" width="270" alt="Example Image" class="d-inline-block mt-3 p-3 rounded img-fluid" /></a>

## 💚 Community & Support

*   ⭐ [GitHub](https://github.com/openlit/openlit/)
*   🌍 [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3)
*   🐞 [GitHub Issues](https://github.com/openlit/openlit/issues)
*   𝕏 [X](https://twitter.com/openlit_io)

## License

OpenLIT is licensed under the [Apache-2.0 license](LICENSE).

## 🙇‍♂️ Acknowledgments

<p>This project is proudly supported by:</p>
<p>
  <a href="https://www.digitalocean.com/">
    <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="201px">
  </a>
</p>
```
Key improvements and SEO optimizations:

*   **Concise Headline:**  Uses the most relevant keywords (AI Engineering, Observability) while being engaging.
*   **Descriptive Subheadings:** Clear section headings for easy navigation.
*   **Bulleted Key Features:**  Highlights the core benefits in a scannable format.
*   **Keyword Optimization:**  Uses terms like "AI Engineering," "LLMs," "Observability," and "Guardrails" strategically.
*   **Strong Call to Action:** Encourages users to get started and contribute.
*   **Clear Steps:**  Provides concise installation instructions.
*   **Community Links:** Makes it easy to engage with the project and ask questions.
*   **Simplified Structure:**  Removes redundant information and improves readability.
*   **Product Hunt & Fazier Badges:**  Includes links to the project's presence on these platforms.
*   **Direct Links:** The provided links now direct users directly to the right pages.