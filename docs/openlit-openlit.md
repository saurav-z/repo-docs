<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: Your Open-Source Platform for AI Engineering

**Simplify your AI development workflow and gain deep insights with OpenLIT, a comprehensive platform for observability, evaluations, and more.**  Explore the power of OpenLIT with our [Documentation](https://docs.openlit.io/), [Quickstart Guide](-getting-started-with-llm-observability), [Python SDK](https://github.com/openlit/openlit/tree/main/sdk/python), and [Typescript SDK](https://github.com/openlit/openlit/tree/main/sdk/typescript).  Check out the [Roadmap](#-roadmap) to see what's planned!

**[Visit the original repository on GitHub](https://github.com/openlit/openlit)**

[![OpenLIT](https://img.shields.io/badge/OpenLIT-orange)](https://openlit.io/)
[![License](https://img.shields.io/github/license/openlit/openlit?label=License&logo=github&color=f80&logoColor=white)](https://github.com/openlit/openlit/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/openlit/month)](https://pepy.tech/project/openlit)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/openlit/openlit)](https://github.com/openlit/openlit/pulse)
[![GitHub Contributors](https://img.shields.io/github/contributors/openlit/openlit)](https://github.com/openlit/openlit/graphs/contributors)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ)
[![X](https://img.shields.io/badge/follow-%40openlit__io-1DA1F2?logo=x&style=social)](https://twitter.com/openlit_io)

---

<https://github.com/user-attachments/assets/6909bf4a-f5b4-4060-bde3-95e91fa36168>

OpenLIT streamlines your AI development, particularly for Generative AI and LLMs. With OpenLIT, you can effortlessly experiment with LLMs, manage prompts effectively, and securely handle API keys. It provides OpenTelemetry-native observability with just one line of code, offering complete monitoring across LLMs, vector databases, and GPUs. Build and deploy AI features with confidence, moving seamlessly from testing to production.  OpenLIT fully supports the [Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai) with the OpenTelemetry community, staying current with Observability standards.

## Key Features of OpenLIT

*   **📈 Analytics Dashboard:**  Visualize AI application health and performance with dashboards for metrics, costs, and user interactions.
*   **🔌 OpenTelemetry-native SDKs:**  Vendor-neutral SDKs to send traces and metrics to your existing observability tools, for unified monitoring.
*   **💲 Cost Tracking:**  Optimize budgets with detailed cost estimations for custom and fine-tuned models using customizable pricing files.
*   **🐛 Exceptions Monitoring:**  Quickly identify and resolve issues with a dedicated dashboard for tracking exceptions and errors.
*   **💭 Prompt Management:**  Organize and version prompts in a Prompt Hub for consistent, easy access across all applications.
*   **🔑 API Key & Secret Management:**  Securely manage API keys and secrets centrally to improve security.
*   **🎮 Experiment with LLMs:**  Use OpenGround to test and compare various LLMs, for better model selection.

## Getting Started with OpenLIT - LLM Observability

Here's how to easily set up OpenLIT and start gaining observability into your LLM applications:

### Step 1: Deploy OpenLIT Stack

1.  **Clone the Repository:**

    ```shell
    git clone git@github.com:openlit/openlit.git
    ```

2.  **Self-host using Docker:**

    ```shell
    docker compose up -d
    ```
    *For Kubernetes installation, see the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes).*

### Step 2: Install OpenLIT SDK

Choose the SDK for your project:

```bash
pip install openlit  # Python SDK
```
*For the TypeScript SDK, please see the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).*

### Step 3: Initialize OpenLIT in Your Application

Integrate OpenLIT into your AI applications by adding the necessary initialization code.

```python
import openlit

openlit.init()
```

Configure the telemetry data destination:

| Purpose                            | Parameter/Environment Variable                   | For Sending to OpenLIT    |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default   |

>   **💡 Note:**  If no `otlp_endpoint` is given, the SDK will output traces to your console during development.

#### Example Code

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

With OpenLIT collecting your observability data, you can start to visualize and analyze this data to improve your AI application.

*   Access your OpenLIT dashboard by going to `127.0.0.1:3000` in your browser.
*   Log in with the default credentials:  **Email:** `user@openlit.io`, **Password:** `openlituser`

![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true)
![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true)

## 🛣️ Roadmap

See what's new and what's coming next:

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

We welcome contributions! See our [Contribution guide](./CONTRIBUTING.md) to get started.

Ways to help:

*   Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) to discuss ideas and give feedback.

## 💚 Community & Support

Stay connected and get help:

*   ⭐ Star the repo on [GitHub](https://github.com/openlit/openlit/).
*   💬 Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3) community for discussions.
*   🐞 Report bugs via our [GitHub Issues](https://github.com/openlit/openlit/issues).
*   🐦 Follow us on [X](https://twitter.com/openlit_io).

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
Key improvements and SEO considerations:

*   **Concise Hook:** The opening sentence is a compelling hook, immediately highlighting the value proposition.
*   **Clear Headings:**  Uses H1, H2, and H3 tags to structure the document, which improves readability and SEO.
*   **Keyword Optimization:** Uses relevant keywords like "AI Engineering," "LLMs," "Observability," and "OpenTelemetry."  Keywords are integrated naturally.
*   **Bulleted Key Features:**  Highlights the core benefits, making it easy for users to scan and understand the value.
*   **Clear Call to Action:** Encourages users to explore the documentation and quickstart.
*   **Links to Important Pages:** Provides direct links to documentation, SDKs, and the repository.
*   **Detailed Getting Started:** Includes step-by-step instructions for setup.
*   **Roadmap Section:** Showcases the project's progress and future plans.
*   **Contribution and Community Sections:**  Provides information on how to contribute and get support, which builds community and engagement.
*   **License Information:** Includes the license for transparency.
*   **Acknowledgments:** Gives proper credit to supporters.
*   **Concise, Readable Language:** Improved the flow and clarity of the text.
*   **Image Optimization:**  Keeps images and logos, which improve aesthetics.