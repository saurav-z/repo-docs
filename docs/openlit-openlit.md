<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: The Open Source Platform for Streamlining AI Engineering and LLM Development

**[OpenLIT](https://github.com/openlit/openlit) empowers developers to build, monitor, and optimize AI applications with ease, providing observability, evaluations, guardrails, and more.**

**Key Features:**

*   📈 **AI Application Performance Analytics:** Monitor health, user interactions, and costs with detailed dashboards.
*   🔌 **OpenTelemetry-Native Observability:** Leverage vendor-neutral SDKs for comprehensive tracing and metric collection.
*   💲 **Cost Tracking for LLMs:**  Tailor cost estimations for specific models using custom pricing files for precise budgeting.
*   🐛 **Exception Monitoring:** Identify and resolve issues quickly with a dedicated monitoring dashboard.
*   💭 **Prompt Management:** Organize and version prompts for consistent use across applications.
*   🔑 **Secure Secrets Management:**  Protect API keys and other sensitive data centrally.
*   🎮 **LLM Experimentation:**  Use OpenGround to test and compare different LLMs side-by-side.
*   **...and more!**

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

### Why Choose OpenLIT?

OpenLIT simplifies AI development by providing the tools you need to experiment, monitor, and optimize your LLM-powered applications, from development to production.  With OpenTelemetry-native observability, you gain full-stack visibility into your AI workflows.

## 🚀 Getting Started with LLM Observability

```mermaid
flowchart TB;
    subgraph " "
        direction LR;
        subgraph " "
            direction LR;
            OpenLIT_SDK[OpenLIT SDK] -->|Sends Traces & Metrics| OTC[OpenTelemetry Collector];
            OTC -->|Stores Data| ClickHouseDB[ClickHouse];
        end
        subgraph " "
            direction RL;
            OpenLIT_UI[OpenLIT] -->|Pulls Data| ClickHouseDB;
        end
    end
```

### Step 1: Deploy OpenLIT Stack

1.  **Clone the OpenLIT Repository:**
    ```shell
    git clone git@github.com:openlit/openlit.git
    ```

2.  **Self-host with Docker:**

    ```shell
    docker compose up -d
    ```

>   *For Kubernetes (Helm), see the [installation guide](https://docs.openlit.io/latest/installation#kubernetes).*

### Step 2: Install the OpenLIT SDK

```bash
pip install openlit
```

>   *See the [TypeScript SDK guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation) for TypeScript.*

### Step 3: Initialize OpenLIT in Your Application

Integrate OpenLIT with a few lines of code.

```python
import openlit

openlit.init()
```

Configure the telemetry data destination:

| Purpose                            | Parameter/Environment Variable                   | For Sending to OpenLIT    |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default   |

>   *By default, the SDK outputs to your console, useful during development.*

#### Example

---

<details>
  <summary>Initialize using Function Arguments</summary>

  Add the following two lines to your application code:

  ```python
  import openlit

  openlit.init(
    otlp_endpoint="http://127.0.0.1:4318",
  )
  ```
</details>

---

<details>

  ---

  <summary>Initialize using Environment Variables</summary>

  Add the following two lines to your application code:

  ```python
  import openlit

  openlit.init()
  ```

  Then, configure the your OTLP endpoint using environment variable:

  ```env
  export OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318"
  ```
</details>

---

### Step 4: Visualize and Optimize

OpenLIT provides insights into your application's performance. Access the OpenLIT UI at `127.0.0.1:3000` in your browser.

**Default Credentials:**
*   **Email:** `user@openlit.io`
*   **Password:** `openlituser`

![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true)
![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true)

## 🛣️ Roadmap

Our development is ongoing. Here are recent and planned features:

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

We welcome contributions!  See our [Contribution guide](./CONTRIBUTING.md) to get started.
Unsure where to start?

-   Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) to discuss ideas.

[![OpenLIT - One click observability, evals for LLMs & GPUs | Product Hunt](https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=460690&theme=light)](https://www.producthunt.com/posts/openlit?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-openlit)
<a href="https://fazier.com/launches/openlit-2" target="_blank" rel="noopener noreferrer"><img src="https://fazier.com/api/v1/public/badges/embed_image.svg?launch_id=779&badge_type=daily" width="270" alt="Example Image" class="d-inline-block mt-3 p-3 rounded img-fluid" /></a>

## 💚 Community & Support

*   ⭐ Star us on [GitHub](https://github.com/openlit/openlit/).
*   🌍 Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3).
*   🐞 Report issues on [GitHub Issues](https://github.com/openlit/openlit/issues).
*   𝕏 Follow us on [X](https://twitter.com/openlit_io).

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

Key improvements and SEO considerations:

*   **Strong Hook:** A concise, benefit-driven introduction to capture attention.
*   **Keywords:** Repeated mentions of key terms like "AI Engineering," "LLMs," "Observability," "Monitoring," "Prompt Management," and "OpenTelemetry" to improve search ranking.
*   **Clear Headings & Structure:** Makes the information easy to scan and digest, improving readability for both users and search engines.
*   **Bulleted Lists:** Highlights key features for quick understanding.
*   **Links:**  Internal and external links to boost SEO and user experience.
*   **Concise Language:** Streamlines text for clarity and better engagement.
*   **Roadmap:** Provides transparency on project direction.
*   **Call to Action:** Encourages community involvement.
*   **Alt Text:** Added to images for accessibility and SEO.