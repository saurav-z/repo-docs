<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">

# OpenLIT: The Open Source Platform for AI Engineering

</div>

**OpenLIT empowers you to build, observe, and optimize your AI applications with ease.**  This platform simplifies AI development workflows, especially for Generative AI and LLMs, providing essential tools for observability, evaluation, and prompt management. Explore key features like dashboards for monitoring performance, cost tracking, and prompt management, and much more!  **[Check out the original repo here!](https://github.com/openlit/openlit)**

*   **[Documentation](https://docs.openlit.io/)** | **[Quickstart](-getting-started-with-llm-observability)** | **[Python SDK](https://github.com/openlit/openlit/tree/main/sdk/python)** | **[Typescript SDK](https://github.com/openlit/openlit/tree/main/sdk/typescript)**

*   **[Roadmap](#️-roadmap)** | **[Feature Request](https://github.com/openlit/openlit/issues/new?assignees=&labels=%3Araised_hand%3A+Up+for+Grabs%2C+%3Arocket%3A+Feature&projects=&template=feature-request.md&title=%5BFeat%5D%3A)** | **[Report a Bug](https://github.com/openlit/openlit/issues/new?assignees=&labels=%3Abug%3A+Bug%2C+%3Araised_hand%3A+Up+for+Grabs&projects=&template=bug.md&title=%5BBug%5D%3A)**

[![OpenLIT](https://img.shields.io/badge/OpenLIT-orange)](https://openlit.io/)
[![License](https://img.shields.io/github/license/openlit/openlit?label=License&logo=github&color=f80&logoColor=white)](https://github.com/openlit/openlit/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/openlit/month)](https://pepy.tech/project/openlit)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/openlit/openlit)](https://github.com/openlit/openlit/pulse)
[![GitHub Contributors](https://img.shields.io/github/contributors/openlit/openlit)](https://github.com/openlit/openlit/graphs/contributors)

[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ)
[![X](https://img.shields.io/badge/follow-%40openlit__io-1DA1F2?logo=x&style=social)](https://twitter.com/openlit_io)

---

## Key Features

*   **Observability**: Gain deep insights with OpenTelemetry-native SDKs for comprehensive tracing and metrics across your AI stack.
*   **LLM Cost Tracking**: Monitor and manage the expenses associated with custom and fine-tuned models.
*   **Exception Monitoring**:  Quickly identify and resolve issues with a dedicated dashboard for tracking errors.
*   **Prompt Hub**: Centralize and version control your prompts for consistent access and management.
*   **Secure API Key Management**:  Use the Vault to securely store and manage API keys.
*   **LLM Experimentation**: Compare and test different LLMs side-by-side with OpenGround.
*   **Analytics Dashboard**: Monitor your AI application's health and performance with detailed dashboards.

## Getting Started: LLM Observability

Follow these simple steps to integrate OpenLIT into your project and start visualizing data:

### Step 1: Deploy OpenLIT Stack

1.  **Clone the Repository:**

    ```shell
    git clone git@github.com:openlit/openlit.git
    ```

2.  **Self-host using Docker:**

    ```shell
    docker compose up -d
    ```

    > For detailed Kubernetes installation instructions, see the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes).

### Step 2: Install OpenLIT SDK

```bash
pip install openlit
```

>   For TypeScript SDK installation, see the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).

### Step 3: Initialize OpenLIT in Your Application

Integrate OpenLIT into your AI applications by adding the following lines to your code.

```python
import openlit

openlit.init()
```

Configure the telemetry data destination as follows:

| Purpose                            | Parameter/Environment Variable                   | For Sending to OpenLIT    |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default   |

>   💡 Info: If the `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` is not provided, the OpenLIT SDK will output traces directly to your console, which is recommended during the development phase.

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

Access your observability data through the OpenLIT UI at `127.0.0.1:3000`. Login with:

*   **Email**: `user@openlit.io`
*   **Password**: `openlituser`

[OpenLIT UI Screenshot 1]
[OpenLIT UI Screenshot 2]

## 🛣️ Roadmap

Stay updated on upcoming features and improvements:

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

We welcome contributions!  Please see our [Contribution guide](./CONTRIBUTING.md) to get started.

## 💚 Community & Support

Connect with the OpenLIT community:

*   ⭐  Star us on [GitHub](https://github.com/openlit/openlit/)
*   💬  Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3)
*   🐞  Report bugs on our [GitHub Issues](https://github.com/openlit/openlit/issues)
*   𝕏  Follow us on [X](https://twitter.com/openlit_io)

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

*   **Clear, concise, and keyword-rich title & introduction.** This immediately tells the reader what OpenLIT is and what it does.
*   **Keyword Use:** Strategically incorporates keywords like "AI Engineering," "LLMs," "Observability," and related terms throughout the document.
*   **Structured Content:** Uses clear headings and subheadings for readability and SEO benefits.
*   **Bulleted Key Features:** Makes it easy for users to quickly understand the core functionality.  More descriptive feature names.
*   **Call to Action:** Encourages exploration with links to key resources (Docs, Quickstart, SDKs)
*   **Strong focus on LLMs**: mentions LLMs early and often.
*   **Concise Getting Started:** Streamlines the installation and initialization steps.
*   **Roadmap:** Clearly outlines planned features.
*   **Community & Support:** Emphasizes community engagement and provides multiple channels for support.
*   **Alt text for Images:** I added alt text for the images for accessibility and SEO.
*   **Overall Readability:** Improved formatting for better user experience and scannability.