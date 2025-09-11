<div align="center">
  <img src="img/banner.svg" alt="The workflow automation platform for security and IT response engineering.">
</div>

</br>

<div align="center">

![Commits](https://img.shields.io/github/commit-activity/m/TracecatHQ/tracecat?style=for-the-badge&logo=github)
![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge&logo=agpl)
[![Discord](https://img.shields.io/discord/1212548097624903681.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/H4XZwsYzY4)

</div>

<div align="center">

<a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xIDFoLTZhMyAzIDAgMCAwLTMgMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
<a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>

</div>

## Tracecat: Automate Security & IT Workflows with Open Source Power

Tracecat is an open-source platform that empowers security and IT engineers to automate workflows, integrate tools, and streamline incident response.  [Explore Tracecat on GitHub](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-Based Templates:** Easily define integrations and workflows using simple, human-readable YAML.
*   **No-Code UI:**  Design and manage workflows with an intuitive, no-code user interface.
*   **Built-in Lookup Tables:** Leverage built-in lookup tables for enriched data and context.
*   **Case Management:** Streamline incident response with integrated case management capabilities.
*   **Reliable Orchestration:** Powered by Temporal for scalable and reliable workflow execution.
*   **Open Cyber Security Schema (OCSF) Alignment:**  Template inputs are normalized to fit the OCSF ontology where possible.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Get up and running quickly with a local Tracecat stack using Docker Compose.  Full instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**(For advanced users):** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Detailed instructions can be found [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry offers a growing collection of integration and response-as-code templates, organized by common capabilities.

*   **Template Library:** Access pre-built templates for various security and IT tasks.
*   **Open Source:** Contribute to and customize templates for your specific needs.
*   **OCSF-Aligned:** Template inputs are normalized to fit the Open Cyber Security Schema (OCSF) ontology where possible.

**Examples:**

*   Visit our documentation on Tracecat Registry for use cases and ideas.
*   Check out existing open source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community & Support

Join the Tracecat community to ask questions, share feedback, and connect with other users and developers!

*   **Discord:**  Join the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

Tracecat is primarily available under the AGPL-3.0 license, with the exception of the `ee` directory, which contains paid enterprise features.  The Enterprise Edition offers advanced features and requires a Tracecat Enterprise license.

*For information about Tracecat's Enterprise self-hosted or managed Cloud offerings, please visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

Tracecat is committed to providing a secure platform. Security features like SSO, audit logs, and IaC deployments are available in the open-source version. We are working on a comprehensive list of the threat model, security features, and hardening recommendations.

*   **Report Security Issues:**  Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) and include `tracecat` in the subject line.

## Contributors

Thank you to all our amazing contributors for their code, integrations, and support!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>
```
Key improvements and explanations:

*   **Concise Hook:**  The one-sentence hook effectively introduces Tracecat and its core purpose.
*   **SEO-Optimized Headings:**  Uses clear and descriptive headings like "Key Features", "Getting Started", "Community & Support", etc., for better search engine indexing.  Includes H2 and H3 tags.
*   **Bulleted Key Features:**  Provides a concise summary of Tracecat's core capabilities, making it easy for users to understand the value proposition.  Uses bullet points for clarity.
*   **Clear Call to Actions:**  Encourages users to explore the project by linking to the GitHub repository, documentation, and Discord community.
*   **Keyword Optimization:** Includes relevant keywords like "security automation", "IT automation", "incident response", "open source", "workflows", and "templates."
*   **Improved Formatting:** Utilizes bold text, and bullet points to enhance readability.
*   **Community Focus:** Highlights the community aspect, encouraging user engagement.
*   **Clear License Information:**  Keeps the license information and contributor images intact for attribution.
*   **Links to GitHub & Documentation:**  Ensures users can easily access the original repo and documentation.
*   **Removed Redundancy:** Combines similar sentences, and removes unnecessary phrases.
*   **Uses existing images:** Keeps the existing images from the original README.
*   **Clear Explanations:** Uses clear language to explain each section.