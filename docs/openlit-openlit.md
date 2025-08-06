<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: The Open Source Platform for Streamlining AI Engineering

OpenLIT simplifies your AI development journey by providing observability, evaluations, guardrails, prompt management, and more.  ([See the original repo](https://github.com/openlit/openlit))

**Key Features:**

*   📈 **AI Application Observability:** Gain real-time insights into your LLM application's performance with detailed dashboards for metrics, costs, and user interactions, built on OpenTelemetry.
*   🔌 **OpenTelemetry-Native SDKs:** Easily integrate with existing observability tools using vendor-neutral SDKs.
*   💲 **LLM Cost Tracking:** Optimize spending with precise cost estimations for custom and fine-tuned models.
*   🐛 **Exception Monitoring:** Quickly identify and resolve issues with a dedicated exceptions dashboard.
*   💭 **Prompt Management:** Organize and version prompts with a centralized Prompt Hub.
*   🔑 **Secure Secrets Management:** Safely handle API keys and sensitive data.
*   🎮 **LLM Experimentation:** Test and compare various LLMs using OpenGround.

## Getting Started

### Step 1: Deploy OpenLIT Stack

1.  **Clone the Repository:**
    ```bash
    git clone git@github.com:openlit/openlit.git
    ```

2.  **Run with Docker:**
    ```bash
    docker compose up -d
    ```

    *For Kubernetes deployment instructions, refer to the [installation guide](https://docs.openlit.io/latest/installation#kubernetes).*

### Step 2: Install OpenLIT SDK

```bash
pip install openlit
```

*For TypeScript SDK installation, visit the [TypeScript SDK Guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).*

### Step 3: Initialize in Your Application

Add the following to your code:

```python
import openlit
openlit.init()
```

Configure the telemetry destination:

| Purpose                            | Parameter/Environment Variable                   | Example               |
| ---------------------------------- | ------------------------------------------------ | --------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |

### Step 4: Visualize and Optimize

Access the OpenLIT UI at `127.0.0.1:3000`.

*   **Login:** `user@openlit.io` / `openlituser`

<img src="https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true" alt="OpenLIT UI 1" width="40%">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true" alt="OpenLIT UI 2" width="40%">

## Roadmap

*   **Completed:** OpenTelemetry-native Observability, GPU Monitoring, Exceptions Monitoring, Prompt Hub, OpenGround, Vault, Cost Tracking, Guardrails, Evaluations.
*   **Coming Soon:** Auto-Evaluation Metrics, Human Feedback, Dataset Generation, Search over Traces.

## Contributing

We welcome contributions!  See our [Contribution guide](./CONTRIBUTING.md) or join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) community.

## Community & Support

*   ⭐ Star us on [GitHub](https://github.com/openlit/openlit/).
*   💬 Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3).
*   🐞 Report bugs on [GitHub Issues](https://github.com/openlit/openlit/issues).
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