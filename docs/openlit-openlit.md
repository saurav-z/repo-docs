<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: The Open Source Platform for AI Engineering

**Simplify your AI development workflow with OpenLIT, the platform designed for observability, evaluations, guardrails, prompt management, and more.**  ([Original Repo](https://github.com/openlit/openlit))

**Key Features:**

*   📈 **AI Application Monitoring:** Monitor your AI applications with detailed dashboards for key metrics, costs, and user interactions.
*   🔌 **OpenTelemetry-native Observability:**  Leverage vendor-neutral SDKs to send traces and metrics to your existing observability tools.
*   💲 **Cost Tracking:** Accurately estimate model costs, including custom and fine-tuned models.
*   🐛 **Exception Monitoring:** Quickly identify and resolve issues with a dedicated dashboard for tracking exceptions and errors.
*   💭 **Prompt Management:** Organize and version prompts with Prompt Hub.
*   🔑 **API Key Security:** Securely manage and centralize API keys and secrets.
*   🎮 **LLM Experimentation:** Easily explore, test, and compare different LLMs.

## Getting Started

### 1. Deploy the OpenLIT Stack

You can deploy OpenLIT using Docker Compose or Kubernetes with Helm.

1.  **Docker Compose (Self-Hosted):**

    ```bash
    git clone git@github.com:openlit/openlit.git
    docker compose up -d
    ```

    > For detailed Kubernetes installation instructions, see the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes)

### 2. Install the OpenLIT SDK

   Choose your preferred SDK:

   *   **Python:**

       ```bash
       pip install openlit
       ```
   *   **TypeScript:** Refer to the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).

### 3. Initialize the SDK in your Application

Import and initialize the OpenLIT SDK in your application code:

```python
import openlit

openlit.init()
```

Configure the telemetry data destination as follows:

| Purpose                            | Parameter/Environment Variable                   | For Sending to OpenLIT    |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default   |

> 💡 If no `otlp_endpoint` is provided, the SDK will output traces to your console.

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

### 4. Visualize & Optimize

Access the OpenLIT UI at `127.0.0.1:3000` in your browser. Use the following credentials:

*   **Email:** `user@openlit.io`
*   **Password:** `openlituser`

Analyze your observability data to gain valuable insights.

![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true)
![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true)

## 🛣️ Roadmap & Updates

Stay informed about OpenLIT's ongoing development:

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

Unsure where to begin?

-   Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) to share ideas, provide feedback, and connect with the OpenLIT community.

## 💚 Community & Support

Connect with us:

*   ⭐ [GitHub](https://github.com/openlit/openlit/) - Star us!
*   💬 [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) and [Discord](https://discord.gg/CQnXwNT3) - Join the community.
*   🐞 [GitHub Issues](https://github.com/openlit/openlit/issues) - Report bugs.
*   𝕏 [X](https://twitter.com/openlit_io) - Follow for updates.

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

*   **Clear, Concise Hook:**  The initial sentence immediately tells the user what OpenLIT is and its main benefits.
*   **Keyword Optimization:** Used keywords like "AI Engineering," "observability," "LLMs," "Prompts," "metrics," and "cost tracking" to improve searchability.
*   **Headings:**  Organized the information into clear, scannable sections using headings (H1, H2, H3) for readability and SEO.
*   **Bulleted Lists:**  Made key features and installation steps easy to digest.
*   **Links:**  Provided links to the original repo, documentation, SDKs, and the community.  Internal links (e.g., to roadmap) are also good.
*   **Conciseness:**  Removed unnecessary repetition and streamlined the language.
*   **Call to Action (CTA):** Encourages users to join the community and contribute.
*   **Roadmap Highlight:** The roadmap section is very important and kept.
*   **Direct, Clear Instructions:**  Installation instructions are focused.
*   **Alt Text for Images:**  Added `alt` text to the logo image for accessibility and SEO.
*   **SEO best practice:** The title is repeated again to improve chances to get search results.