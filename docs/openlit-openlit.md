<div align="center">
<img src="https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true" alt="OpenLIT Logo" width="30%">
</div>

# OpenLIT: Your All-in-One Platform for AI Engineering

**OpenLIT simplifies and accelerates your AI development workflow, providing powerful tools for LLM observability, prompt management, and more.**  [Explore the OpenLIT repository](https://github.com/openlit/openlit) for more details.

**Key Features:**

*   📈 **AI Observability:** Gain real-time insights into your AI application's performance with OpenTelemetry-native SDKs.
*   💭 **Prompt Management:** Version, organize, and share prompts efficiently with Prompt Hub.
*   🔑 **Secure Secrets Management:** Protect API keys and sensitive information using our built-in Vault.
*   💲 **Cost Tracking:** Get precise cost estimations for custom and fine-tuned models.
*   🎮 **LLM Experimentation:** Easily test and compare different LLMs with OpenGround.
*   🐛 **Exception Monitoring:** Quickly identify and resolve errors with a dedicated dashboard.
*   ⚡️ **Real-Time Guardrails:** Ensure safety and compliance of your AI applications.
*   🔬 **LLM Evaluations:** Automate performance testing based on usage with our programmatic evaluations.

## Getting Started

Quickly integrate OpenLIT into your project for comprehensive AI observability:

### 1. Deploy OpenLIT Stack

*   **Clone the repository:**
    ```bash
    git clone git@github.com:openlit/openlit.git
    ```
*   **Self-host using Docker:**
    ```bash
    docker compose up -d
    ```
    > For Kubernetes installation instructions, refer to the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes).

### 2. Install OpenLIT SDK

```bash
pip install openlit
```
> For the TypeScript SDK, see the [TypeScript SDK Installation guide](https://github.com/openlit/openlit/tree/main/sdk/typescript#-installation).

### 3. Initialize in Your Application

Add the following code to your application:

```python
import openlit

openlit.init()
```
Configure the telemetry endpoint.

| Purpose                            | Parameter/Environment Variable                   | For Sending to OpenLIT    |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Send data to an HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Authenticate telemetry backends    | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default   |

> 💡 Tip:  During development, the OpenLIT SDK defaults to console output if no endpoint is set.

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

  ---

  <summary>Initialize using Environment Variables</summary>

  ```python
  import openlit

  openlit.init()
  ```

  ```env
  export OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318"
  ```

</details>

---

### 4. Visualize and Optimize

Access the OpenLIT dashboard at `127.0.0.1:3000` to view your observability data.

*   **Login:**
    *   Email: `user@openlit.io`
    *   Password: `openlituser`

   ![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true)
   ![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-2.png?raw=true)

## Roadmap

Stay updated on our progress:

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

We welcome contributions!  Review our [Contribution guide](./CONTRIBUTING.md) to get started.

Connect with us:

*   Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/rjvTm6zd) community.
*   Report bugs on our [GitHub Issues](https://github.com/openlit/openlit/issues).

<a href="https://www.producthunt.com/posts/openlit?utm_source=badge-featured&utm_medium=badge&utm_souce=badge-openlit">
  <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=460690&theme=light" alt="OpenLIT - One click observability, evals for LLMs & GPUs | Product Hunt" />
</a>
<a href="https://fazier.com/launches/openlit-2" target="_blank" rel="noopener noreferrer"><img src="https://fazier.com/api/v1/public/badges/embed_image.svg?launch_id=779&badge_type=daily" width="270" alt="Example Image" class="d-inline-block mt-3 p-3 rounded img-fluid" /></a>

## Community and Support

*   ⭐ Give us a star on [GitHub](https://github.com/openlit/openlit/).
*   💬 Join our [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or [Discord](https://discord.gg/CQnXwNT3) for live discussions.
*   🐞 Report bugs on [GitHub Issues](https://github.com/openlit/openlit/issues).
*   🐦 Follow us on [X](https://twitter.com/openlit_io).

## License

OpenLIT is licensed under the [Apache-2.0 license](LICENSE).

## Acknowledgments

<p>This project is proudly supported by:</p>
<p>
  <a href="https://www.digitalocean.com/">
    <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="201px">
  </a>
</p>
```
Key improvements and optimizations:

*   **SEO-Friendly Title & Description:** Includes the primary keywords "AI Engineering," "LLM Observability," and related terms early in the title and description, making it more discoverable.  The initial sentence acts as a hook, summarizing the project's core value.
*   **Clear Headings:** Uses Markdown headings (H1, H2, etc.) to organize content logically, enhancing readability and SEO.
*   **Concise Bullet Points:** Presents key features in an easily digestible bulleted list, improving user comprehension and SEO.
*   **Action-Oriented Language:** Uses verbs like "simplify," "accelerate," and "explore" to encourage engagement.
*   **Direct Links:** Provides links to the original repo.
*   **Focused Introduction:**  The intro directly states what OpenLIT does (focusing on benefits) and emphasizes the ease of getting started.
*   **Simplified Getting Started:**  Steps are clear and easy to follow, making it very user-friendly.  Uses code blocks for easy copy-pasting.
*   **Clear Roadmap:** Uses a table for easy scanning and understanding of the project's future.
*   **Community & Support:** Sections are clear and encourage community participation and support.
*   **Cleaned-up formatting:** Removed unnecessary code/styling, streamlining the README.
*   **Keywords:** Added relevant keywords naturally throughout the text.
*   **Removed redundant images:** The repeated banner image was removed.
*   **Replaced Mermaid diagram with text:** Mermaid diagrams are useful, but often render strangely. This version removes this for ease of use.