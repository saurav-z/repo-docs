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

<a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xMWgtNmEzIDMgMCAwIDAtMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
<a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>

</div>

## Tracecat: Automate Security and IT Response with Ease

Tracecat is a modern, open-source workflow automation platform empowering security and IT engineers to streamline incident response and operational tasks.  [View the code on GitHub](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-Based Templates:** Define integrations and workflows simply using human-readable YAML.
*   **No-Code UI:** Build and manage workflows easily with an intuitive visual interface.
*   **Built-in Lookup Tables & Case Management:** Enhance investigations and automate tasks with integrated data and case handling.
*   **Reliable Orchestration:** Built on Temporal for scalable and dependable workflow execution.
*   **Tracecat Registry:** Access a curated library of response-as-code templates and integrations.
*   **Open Cyber Security Schema (OCSF) Alignment:** Utilize OCSF to normalize and standardize input/output, for greater interoperability.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform.  Instructions can be found [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat Community Discord to connect with other users, ask questions, and share new integration ideas.  [Join the Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry provides a centralized hub of integration and response-as-code templates to streamline your security operations.

**Key Benefits of the Registry:**

*   **Modular Actions:** Response actions are organized by common capabilities (e.g., `list_alerts`, `list_cases`).
*   **OCSF Compatibility:** Template inputs are normalized to the Open Cyber Security Schema (OCSF) where possible.
*   **Extensive Library:** Access a growing collection of pre-built integrations and automation templates.

Explore use cases and template ideas in our documentation.  Check out existing open source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

Tracecat is primarily available under the AGPL-3.0 license. Enterprise features are available in the `ee` directory and require a paid license.

The Enterprise Edition offers additional features requiring specialized investments in research and development.  You can enable the Enterprise Edition directly in the settings of the platform.

For information about Tracecat's Enterprise self-hosted or managed Cloud offering, please visit [our website](https://tracecat.com) or [book a meeting with us](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security. Free features include SSO, audit logs, and infrastructure-as-code (IaC) deployments (Terraform, Kubernetes / Helm). We are developing a comprehensive security model, threat model, and hardening recommendations. Contact us on [Discord](https://discord.gg/H4XZwsYzY4) for immediate security questions.

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## Contributors

Thank you to our contributors for their code, integrations, and support! ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">
  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>
</div>