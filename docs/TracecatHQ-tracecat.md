<div align="center">
  <img src="img/banner.svg" alt="The workflow automation platform for security and IT response engineering.">
</div>

<br>

<div align="center">
  <!-- Badges -->
  ![Commits](https://img.shields.io/github/commit-activity/m/TracecatHQ/tracecat?style=for-the-badge&logo=github)
  ![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge&logo=agpl)
  [![Discord](https://img.shields.io/discord/1212548097624903681.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/H4XZwsYzY4)
  <br>
  <a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xMWgtNmEzIDMgMCAwIDAtMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
  <a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>
</div>

## Tracecat: Automate Security & IT Workflows with Open Source

Tracecat is an open-source platform empowering security and IT engineers to automate their workflows with ease.  [Check out the original repo](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-Based Templates:** Define integrations and workflows using simple, human-readable YAML.
*   **No-Code UI:** Build and manage workflows through an intuitive user interface.
*   **Built-in Lookup Tables & Case Management:** Streamline investigations and incident response.
*   **Scalable & Reliable:** Orchestrated with Temporal for high performance and resilience.
*   **Open Cyber Security Schema (OCSF) Integration:** Templates are normalized to align with the OCSF for consistency.

![Tracecat Workflow Example](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is under active development and may have breaking changes between releases.  Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy Tracecat locally using Docker Compose.  Find complete instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Full instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community for support, feedback, and new integration ideas! Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry is a curated collection of integration templates, offering a library of response actions organized by capability (e.g., `list_alerts`, `list_cases`).

**Key Benefits:**

*   **Pre-built Integrations:** Leverage pre-built templates for common security and IT tasks.
*   **Standardized Inputs:** Template inputs are normalized to the Open Cyber Security Schema (OCSF) for consistency.

**Examples & Resources:**

*   Explore use cases and ideas in our documentation.
*   Browse existing open-source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

Tracecat is primarily available under the AGPL-3.0 license, with the exception of the `ee` directory, which contains features requiring a Tracecat Enterprise license. The Enterprise Edition offers advanced capabilities that require specific R&D investments.  You can enable the Enterprise Edition in the platform settings.

*   For information about Tracecat's Enterprise self-hosted or managed Cloud offerings, visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security, offering features such as SSO, audit logs, and IaaC deployments (Terraform, Kubernetes / Helm) for free. We are developing a comprehensive security overview.  For immediate security inquiries, contact us on [Discord](https://discord.gg/H4XZwsYzY4).

Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with "tracecat" in the subject line.

## Contributors

Thank you to all contributors for your code, integrations, and support! ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">
  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>
</div>