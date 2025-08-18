<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: Your Open-Source Platform for AI Engineering

**OpenLIT empowers you to build, monitor, and optimize your AI applications with ease.** This all-in-one platform provides observability, evaluations, guardrails, prompt management, a secure vault, and a playground.  Learn more at the [OpenLIT GitHub Repository](https://github.com/openlit/openlit).

**Key Features:**

*   📈 **AI Application Analytics Dashboard**: Gain insights into your AI application's performance with comprehensive dashboards.
*   🔌 **OpenTelemetry-native Observability SDKs**: Integrate seamlessly with your existing monitoring tools using vendor-neutral SDKs.
*   💲 **Cost Tracking for Custom and Fine-Tuned Models**: Manage your budget effectively with precise cost estimations.
*   🐛 **Exceptions Monitoring Dashboard**: Quickly identify and resolve issues with a dedicated error monitoring dashboard.
*   💭 **Prompt Management**: Organize and version prompts with Prompt Hub for consistent application access.
*   🔑 **API Keys and Secrets Management**: Securely manage your sensitive information centrally.
*   🎮 **Experiment with different LLMs**: Explore, test, and compare various LLMs side by side with OpenGround.
*   🛡️ **Guardrails**: Implement real-time guardrails to ensure responsible AI usage.

## Getting Started

### 1. Deploy OpenLIT Stack

1.  **Clone the repository:**

    ```shell
    git clone git@github.com:openlit/openlit.git
    ```

2.  **Self-host using Docker:**

    ```shell
    docker compose up -d
    ```

    *For Kubernetes deployment using Helm, refer to the [installation guide](https://docs.openlit.io/latest/installation#kubernetes).*

### 2. Install OpenLIT SDK

```bash
pip install openlit
```

*For the TypeScript SDK, see the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).*

### 3. Initialize OpenLIT in Your Application

Add the following lines to your application:

```python
import openlit
openlit.init()
```

Configure the telemetry data destination using either `otlp_endpoint` or environment variables:

| Purpose                            | Parameter/Environment Variable                   | For Sending to OpenLIT    |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default   |

*If no `otlp_endpoint` is provided, the SDK outputs traces to your console (recommended for development).*

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
  
  Then, configure your OTLP endpoint using environment variable:

  ```env
  export OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318"
  ```

</details>

---

### 4. Visualize and Optimize

Access your data at `127.0.0.1:3000` in your browser. Use the default credentials:

*   **Email**: `user@openlit.io`
*   **Password**: `openlituser`

<div align="center">
    <img src="https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true" alt="OpenLIT Dashboard 1" width="45%">
    <img src="https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true" alt="OpenLIT Dashboard 2" width="45%">
</div>

## Roadmap

OpenLIT is constantly evolving.  Here's a look at current and upcoming features:

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

## Contribute

Your contributions are welcome! See the [Contribution guide](./CONTRIBUTING.md) for details.

## Community & Support

*   ⭐ Star the repository on [GitHub](https://github.com/openlit/openlit/).
*   💬 Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3) communities.
*   🐞 Report bugs on [GitHub Issues](https://github.com/openlit/openlit/issues).
*   𝕏 Follow us on [X](https://twitter.com/openlit_io).

## License

OpenLIT is licensed under the [Apache-2.0 license](LICENSE).

## Acknowledgments

<p>This project is proudly supported by:</p>
<p>
  <a href="https://www.digitalocean.com/">
    <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="201px">
  </a>
</p>