<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: The Open Source Platform for AI Engineering 

**[OpenLIT](https://github.com/openlit/openlit) simplifies AI development by providing observability, evaluation tools, prompt management, and secure secrets handling.**

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

OpenLIT helps AI engineers to build production-ready AI applications with greater speed and confidence. From LLM experimentation to comprehensive observability, OpenLIT provides the tools you need to simplify and streamline your AI development workflow.

## Key Features

*   **Comprehensive Observability:**  Leverage OpenTelemetry-native SDKs for full-stack monitoring, including LLMs, vector databases, and GPUs.  Gain insights into your AI application's health and performance with detailed dashboards.
*   **Cost Tracking:**  Accurately track costs for custom and fine-tuned models using custom pricing files for precise budgeting.
*   **Prompt Management:**  Manage and version prompts through Prompt Hub for consistent and easy access across all your applications.
*   **Secure Secrets Management:**  Safely handle API keys and other sensitive information.
*   **LLM Experimentation:** Use OpenGround to explore, test, and compare various LLMs.
*   **Exceptions Monitoring:** Quickly identify and resolve issues with a dedicated monitoring dashboard for exceptions and errors.
*   **Guardrails and Evaluations:** Ensure the safety and performance of your models through configurable guardrails and detailed evaluations.

## Getting Started

To get started with OpenLIT, you can follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone git@github.com:openlit/openlit.git
    ```

2.  **Deploy OpenLIT (Docker Compose):**

    ```bash
    docker compose up -d
    ```

    *For Kubernetes installation, see the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes).*

3.  **Install the Python SDK:**

    ```bash
    pip install openlit
    ```

    *For the TypeScript SDK, see the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).*

4.  **Initialize the SDK in Your Application:**

    ```python
    import openlit
    openlit.init()
    ```

    *   Configure the `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable to point to your OpenLIT instance (e.g., `"http://127.0.0.1:4318"`).
    *   If no endpoint is provided, the SDK will output traces to the console.

    **Example:**

    ```python
    import openlit
    openlit.init(
      otlp_endpoint="http://127.0.0.1:4318", 
    )
    ```

5.  **Visualize and Optimize:**

    Access the OpenLIT UI at `127.0.0.1:3000` in your browser and log in with:

    *   **Email:** `user@openlit.io`
    *   **Password:** `openlituser`

    ![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true)
    ![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true)

## Roadmap

OpenLIT is actively developed. Here's a look at completed and upcoming features:

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

We welcome contributions of all sizes. Please review our [Contribution guide](./CONTRIBUTING.md) for details.

*   Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) communities to discuss ideas.

[![OpenLIT - One click observability, evals for LLMs & GPUs | Product Hunt](https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=460690&theme=light)](https://www.producthunt.com/posts/openlit?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-openlit)
<a href="https://fazier.com/launches/openlit-2" target="_blank" rel="noopener noreferrer"><img src="https://fazier.com/api/v1/public/badges/embed_image.svg?launch_id=779&badge_type=daily" width="270" alt="Example Image" class="d-inline-block mt-3 p-3 rounded img-fluid" /></a>

## Community & Support

*   ⭐ Give us a star on [GitHub](https://github.com/openlit/openlit/).
*   💬 Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3) communities.
*   🐞 Report bugs on our [GitHub Issues](https://github.com/openlit/openlit/issues).
*   𝕏 Follow us on [X](https://twitter.com/openlit_io).

## License

OpenLIT is available under the [Apache-2.0 license](LICENSE).

## Acknowledgments

<p>This project is proudly supported by:</p>
<p>
  <a href="https://www.digitalocean.com/">
    <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="201px">
  </a>
</p>
```
Key improvements:

*   **SEO Optimization:** Added the keywords "AI Engineering," "LLMs," "Observability," and "Open Source" to the title and throughout the description.
*   **Hook:** Added a strong one-sentence hook at the beginning to grab the reader's attention.
*   **Clear Headings:** Improved heading structure for readability and SEO.
*   **Bulleted Key Features:**  Highlights the core functionality.
*   **Concise Language:** Streamlined descriptions.
*   **Call to Action (Community & Support):** Encourages users to engage.
*   **Clearer Structure:** Improved overall formatting and flow.
*   **Links Back to Original Repo:** Includes the links you provided.