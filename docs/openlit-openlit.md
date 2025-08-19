<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: The Open Source Platform for AI Engineering

**OpenLIT empowers AI engineers to build, monitor, and optimize LLM applications with ease.**  

[**View the original repo on GitHub**](https://github.com/openlit/openlit)

**Key Features:**

*   📈 **AI Application Observability:** Monitor AI application health, performance, and user interactions with detailed dashboards.
*   🔌 **OpenTelemetry-Native SDKs:** Vendor-neutral SDKs for seamless integration with existing observability tools.
*   💲 **LLM Cost Tracking:** Precisely track and manage costs for custom and fine-tuned models using custom pricing files.
*   🐛 **Exception Monitoring:** Quickly identify and resolve issues with a dedicated exceptions monitoring dashboard.
*   💭 **Prompt Management:** Version and manage prompts effectively using Prompt Hub.
*   🔑 **Secure API Key Management:** Centralize and securely manage API keys and secrets.
*   🎮 **LLM Experimentation:** Explore, test, and compare different LLMs side-by-side with OpenGround.
*   🛡️ **Guardrails:** Built-in guardrails to help with safety and reliability.
*   ✅ **Evaluations:** Programmatic LLM evaluations for quality assessment.

## 🚀 Get Started with OpenLIT

OpenLIT simplifies AI development with tools for LLM observability, evaluation, and more. Here's how to quickly get started:

### Step 1: Deploy OpenLIT Stack

1.  **Clone the OpenLIT Repository:**
    ```shell
    git clone git@github.com:openlit/openlit.git
    ```
2.  **Self-host using Docker:**
    ```shell
    docker compose up -d
    ```
    > For Kubernetes installation using Helm, see the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes).

### Step 2: Install the OpenLIT SDK

Install the OpenLIT Python SDK:

```bash
pip install openlit
```

>  For TypeScript SDK installation, see the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).

### Step 3: Initialize OpenLIT in Your Application

Integrate OpenLIT into your AI applications with these lines of code:

```python
import openlit

openlit.init()
```

Configure the telemetry data destination:

| Purpose                            | Parameter/Environment Variable                   | For Sending to OpenLIT    |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default   |

> 💡 Info: If `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` is not provided, traces are output to the console during development.

#### Example

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

  Set your OTLP endpoint:

  ```env
  export OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318"
  ```
</details>

---

### Step 4: Visualize and Optimize

Access your OpenLIT instance at `127.0.0.1:3000` in your browser.

*   **Email:** `user@openlit.io`
*   **Password:** `openlituser`

<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true" alt="OpenLIT Dashboard 1" width="45%">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true" alt="OpenLIT Dashboard 2" width="45%">
</div>

## 🛣️ Roadmap

Stay up-to-date on the latest features and enhancements:

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

## 🌱 Contribute

We welcome contributions!  See our [Contribution guide](./CONTRIBUTING.md) to get started.  Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) community.

[![OpenLIT - One click observability, evals for LLMs & GPUs | Product Hunt](https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=460690&theme=light)](https://www.producthunt.com/posts/openlit?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-openlit)
<a href="https://fazier.com/launches/openlit-2" target="_blank" rel="noopener noreferrer"><img src="https://fazier.com/api/v1/public/badges/embed_image.svg?launch_id=779&badge_type=daily" width="270" alt="Example Image" class="d-inline-block mt-3 p-3 rounded img-fluid" /></a>

## 💚 Community & Support

*   🌟 Star us on [GitHub](https://github.com/openlit/openlit/).
*   💬 Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3) community for support.
*   🐞 Report bugs on [GitHub Issues](https://github.com/openlit/openlit/issues).
*   𝕏 Follow us on [X](https://twitter.com/openlit_io).

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
Key improvements and explanations:

*   **SEO-Optimized Title and Introduction:** The title includes the core keywords (OpenLIT, AI Engineering, LLMs, Observability). The introduction now immediately presents the value proposition and uses the target keywords.
*   **Concise One-Sentence Hook:**  The hook clearly states the benefit of using OpenLIT.
*   **Clear Headings and Structure:**  The document is logically organized with clear headings and subheadings for readability and SEO.
*   **Bulleted Key Features:** The features are presented in an easy-to-scan bulleted list. This is crucial for both user experience and search engine indexing.  Relevant keywords are used within the bullet points.
*   **Action-Oriented "Getting Started" Section:** The "Get Started" section is streamlined, providing step-by-step instructions with code examples.
*   **Links:**  All relevant links (documentation, SDKs, community, social media) are included.  The link to the original repo is prominently featured.
*   **Roadmap:** The roadmap is clear and uses emojis to indicate status.
*   **Community and Support:** Section for community interaction and support.
*   **Concise and Clear Language:** The language is more direct and focused, avoiding unnecessary jargon.
*   **Visuals:** Kept the included visuals.
*   **Removed the Mermaid Diagram:** Removed this, as it is a bit complex and, while useful, doesn't contribute directly to the main goals of the document (SEO, clear explanations). If you have a different visualization of the process, replace it.
*   **Product Hunt & Fazier Badges:** Added these to give a professional touch and promote your work.
*   **Emphasis on Value:**  Highlights the benefits for the user (e.g., "Build, Monitor, and Optimize").