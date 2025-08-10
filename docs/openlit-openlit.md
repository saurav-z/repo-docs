<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: The Open Source Platform for AI Engineering

**Simplify and accelerate your AI development journey with OpenLIT, an open-source platform designed to streamline your AI engineering workflow.**

**[Documentation](https://docs.openlit.io/) | [Quickstart](-getting-started-with-llm-observability) | [Python SDK](https://github.com/openlit/openlit/tree/main/sdk/python) | [Typescript SDK](https://github.com/openlit/openlit/tree/main/sdk/typescript) |**

**[Roadmap](#️-roadmap) | [Feature Request](https://github.com/openlit/openlit/issues/new?assignees=&labels=%3Araised_hand%3A+Up+for+Grabs%2C+%3Arocket%3A+Feature&projects=&template=feature-request.md&title=%5BFeat%5D%3A) | [Report a Bug](https://github.com/openlit/openlit/issues/new?assignees=&labels=%3Abug%3A+Bug%2C+%3Araised_hand%3A+Up+for+Grabs&projects=&template=bug.md&title=%5BBug%5D%3A)**

[![OpenLIT](https://img.shields.io/badge/OpenLIT-orange)](https://openlit.io/)
[![License](https://img.shields.io/github/license/openlit/openlit?label=License&logo=github&color=f80&logoColor=white)](https://github.com/openlit/openlit/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/openlit/month)](https://pepy.tech/project/openlit)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/openlit/openlit)](https://github.com/openlit/openlit/pulse)
[![GitHub Contributors](https://img.shields.io/github/contributors/openlit/openlit)](https://github.com/openlit/openlit/graphs/contributors)

[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ)
[![X](https://img.shields.io/badge/follow-%40openlit__io-1DA1F2?logo=x&style=social)](https://twitter.com/openlit_io)

---

<https://github.com/user-attachments/assets/6909bf4a-f5b4-4060-bde3-95e91fa36168>

## Key Features

OpenLIT provides a comprehensive suite of tools to help you build, monitor, and optimize your AI applications, especially those leveraging Generative AI and LLMs.  Here's a look at what OpenLIT has to offer:

*   **Observability:** Gain deep insights into your AI application's health and performance.  Monitor metrics, costs, and user interactions via detailed dashboards.  OpenTelemetry-native SDKs allow integration with existing observability tools.
*   **Cost Management:** Track and optimize costs for custom and fine-tuned LLM models.  Utilize custom pricing files for accurate budgeting.
*   **Error Monitoring:** Identify and resolve issues quickly with a dedicated dashboard for exception and error tracking.
*   **Prompt Management:** Organize and version prompts with Prompt Hub for consistent access and streamlined application development.
*   **Security:** Securely manage API keys and secrets to prevent vulnerabilities.
*   **Experimentation:** Easily test and compare different LLMs with OpenGround.
*   **Guardrails:** Real-time guardrails implementation.
*   **Evaluations:** Programmatic evaluation for LLM responses.

**[Explore the OpenLIT repository on GitHub](https://github.com/openlit/openlit)**

## Getting Started with OpenLIT

This quickstart guide helps you get started with LLM observability.

### 1. Deploy the OpenLIT Stack

1.  **Clone the Repository:**

    ```bash
    git clone git@github.com:openlit/openlit.git
    ```

2.  **Self-host with Docker:**

    ```bash
    docker compose up -d
    ```

    > For Kubernetes installation using Helm, see the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes).

### 2. Install the OpenLIT SDK

```bash
pip install openlit
```

> For the TypeScript SDK, see the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).

### 3. Initialize OpenLIT in Your Application

Integrate OpenLIT into your AI applications with the following lines of code.

```python
import openlit

openlit.init()
```

Configure the telemetry data destination:

| Purpose                            | Parameter/Environment Variable                   | For Sending to OpenLIT    |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default   |

> 💡 Info: If the `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` is not provided, the OpenLIT SDK will output traces directly to your console, which is recommended during the development phase.

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

### 4. Visualize and Optimize

Access OpenLIT at `127.0.0.1:3000` in your browser to explore your observability data. Log in with:

*   **Email:** `user@openlit.io`
*   **Password:** `openlituser`

![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true)
![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true)

## Roadmap

Track OpenLIT's progress and future developments:

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

## Contributing

Help us build the future of AI engineering!  See our [Contribution guide](./CONTRIBUTING.md) to get started.

*   Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) communities to discuss ideas and share feedback.

[![OpenLIT - One click observability, evals for LLMs & GPUs | Product Hunt](https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=460690&theme=light)](https://www.producthunt.com/posts/openlit?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-openlit)
<a href="https://fazier.com/launches/openlit-2" target="_blank" rel="noopener noreferrer"><img src="https://fazier.com/api/v1/public/badges/embed_image.svg?launch_id=779&badge_type=daily" width="270" alt="Example Image" class="d-inline-block mt-3 p-3 rounded img-fluid" /></a>

## Community & Support

Connect with the OpenLIT community:

*   ⭐ Star our [GitHub](https://github.com/openlit/openlit/) repository.
*   💬 Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3) for live discussions.
*   🐞 Report bugs on [GitHub Issues](https://github.com/openlit/openlit/issues).
*   𝕏 Follow us on [X](https://twitter.com/openlit_io) for updates.

## License

OpenLIT is available under the [Apache-2.0 license](LICENSE).

## 🙇‍♂️ Acknowledgments

<p>This project is proudly supported by:</p>
<p>
  <a href="https://www.digitalocean.com/">
    <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="201px">
  </a>
</p>
```
Key improvements and SEO optimizations:

*   **Strong Headline:**  Uses "OpenLIT: The Open Source Platform for AI Engineering" as the main title and a strong hook.
*   **Clear Introduction:**  Provides a concise and engaging introduction highlighting the platform's core value proposition.
*   **SEO Keywords:** The text includes relevant keywords like "AI Engineering," "LLMs," "Observability," "Open Source," "Generative AI," "Monitoring," and "Prompt Management."
*   **Bulleted Key Features:** Presents features in an easy-to-scan bulleted list, emphasizing benefits.
*   **Subheadings:** Uses subheadings to structure the README for readability and SEO.
*   **Clear Calls to Action:** Includes direct links to documentation, SDKs, and other resources.
*   **GitHub Link:** The GitHub link is now prominent in the introduction.
*   **Concise Instructions:** Provides clear and simplified getting started steps.
*   **Community Engagement:** Highlights ways to connect with the community for support and collaboration.
*   **Clean Formatting:**  Uses consistent Markdown formatting for readability.
*   **Product Hunt and Fazier Badges are retained**