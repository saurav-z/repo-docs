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

# Tracecat: Automate Security and IT Workflows with Open Source

**Tracecat is an open-source platform designed to automate security and IT workflows, improving incident response and overall efficiency.**  Check out the original repository [here](https://github.com/TracecatHQ/tracecat).

## Key Features

*   **YAML-Based Templates:** Easily create integrations with simple YAML templates.
*   **No-Code UI:**  A user-friendly interface for designing and managing workflows.
*   **Built-in Lookup Tables:** Simplify data management within your automation processes.
*   **Case Management:**  Streamline incident handling and tracking.
*   **Scalable Architecture:**  Powered by Temporal for reliability and performance.
*   **Open Cyber Security Schema (OCSF) Compatibility:** Integrations designed to align with the OCSF ontology where possible.
*   **Community-Driven:** Join our Discord and collaborate on new integrations!

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Get started quickly with Docker Compose.  Detailed instructions are available in our [documentation](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (Advanced)

Deploy a production-ready Tracecat instance on AWS Fargate using Terraform.  See the full guide [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming Soon!

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

Tracecat Registry is a curated collection of integration and response-as-code templates, organized around common security and IT capabilities.

**Key Benefits:**

*   **Pre-built Integrations:** Jumpstart your automation with ready-to-use templates.
*   **Standardized Actions:**  Templates organized by common capabilities (e.g., `list_alerts`).
*   **OCSF Alignment:** Template inputs are normalized where possible to OCSF standards.

**Explore the Registry:**

*   Visit our documentation for use cases and ideas.
*   Browse existing open source templates in our [repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

Tracecat is primarily available under the AGPL-3.0 license, excluding the `ee` directory which contains enterprise features. The Enterprise Edition provides advanced capabilities and is available via a Tracecat Enterprise license.  Learn more on our [website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat is committed to providing a secure platform. We provide features like SSO and audit logs, along with support for Infrastructure as Code (IaC) deployments. For specific security inquiries, please reach out on [Discord](https://discord.gg/H4XZwsYzY4).

**Security Reporting:** Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including `tracecat` in the subject line.

## Community & Contributions

We appreciate the contributions of our community!  Thank you to everyone who has contributed code, integrations, and support.

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>