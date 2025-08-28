<div align="center">
  <img src="img/banner.svg" alt="The workflow automation platform for security and IT response engineering.">
</div>

</br>

<div align="center">

[![Commits](https://img.shields.io/github/commit-activity/m/TracecatHQ/tracecat?style=for-the-badge&logo=github)](https://github.com/TracecatHQ/tracecat)
[![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge&logo=agpl)](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1212548097624903681.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/H4XZwsYzY4)

</div>

<div align="center">

<a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xIDFoLTZhMyAzIDAgMCAwLTMgMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
<a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>

</div>

## Tracecat: Automate Security & IT Workflows with Open Source Power

Tracecat is a modern, open-source platform designed to streamline security and IT response engineering, offering a powerful way to automate your workflows.  [Explore Tracecat on GitHub](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-Based Templates:** Define integrations and workflows with simple, YAML-based templates for easy configuration.
*   **No-Code UI:** Leverage a user-friendly, no-code UI for intuitive workflow creation and management.
*   **Built-in Lookup Tables & Case Management:**  Enhance investigations and incident response with integrated lookup tables and case management capabilities.
*   **Scalable Orchestration:** Powered by Temporal for reliable, scalable, and resilient workflow execution.
*   **Open Cyber Security Schema (OCSF) Alignment:** Standardize data and improve interoperability through alignment with the OCSF ontology.

![Tracecat workflow](/img/workflow.png)

## Getting Started

Tracecat is in active development, and breaking changes may occur with releases. Always review the [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Locally

1.  **Docker Compose:** Deploy a local Tracecat stack quickly using Docker Compose. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run on Cloud Platforms

1.  **AWS Fargate (Advanced):** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform.  Instructions can be found [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).
2.  **Kubernetes (Coming Soon):**  Stay tuned for Kubernetes deployment options.

## Tracecat Registry: Integration Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry offers a collection of integration and response-as-code templates, organized by common capabilities for streamlined security and IT operations.

**Key Features:**

*   **Action-Oriented Templates:**  Templates are organized around a core set of action capabilities (e.g., `list_alerts`, `list_cases`).
*   **OCSF Data Alignment:** Input parameters are normalized to the Open Cyber Security Schema (OCSF) ontology where possible for improved data standardization.

**Explore the Registry:**

*   **Use Cases and Ideas:** Visit our documentation for use cases and inspiration.
*   **Open Source Templates:**  Explore existing open-source templates within our [GitHub repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is licensed under the AGPL-3.0 license (excluding the `ee` directory, which contains paid enterprise features).

The Enterprise Edition provides advanced features, developed with specific R&D investments. You can enable the Enterprise Edition directly within the platform's settings.

Interested in Tracecat's Enterprise self-hosted or managed Cloud offering? Visit [tracecat.com](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security with features like SSO, audit logs, and Infrastructure-as-Code (IaaC) deployments (Terraform, Kubernetes/Helm), all available for free.  We're actively developing a comprehensive threat model, security features, and hardening recommendations. For immediate security-related questions, please connect with us on [Discord](https://discord.gg/H4XZwsYzY4).

Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including `tracecat` in the subject line.

## Community & Contributing

Join the Tracecat community!  Share questions, feedback, and integration ideas on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

We appreciate our amazing contributors! ❤️  Thank you for your code, integrations, and support.

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>