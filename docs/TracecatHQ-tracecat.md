<div align="center">
  <img src="img/banner.svg" alt="The workflow automation platform for security and IT response engineering.">
</div>

<br>

<div align="center">
  <a href="https://github.com/TracecatHQ/tracecat">
    <img src="https://img.shields.io/github/stars/TracecatHQ/tracecat?style=for-the-badge&logo=github" alt="GitHub stars">
  </a>
  <img src="https://img.shields.io/github/commit-activity/m/TracecatHQ/tracecat?style=for-the-badge&logo=github" alt="Commits">
  <img src="https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge&logo=agpl" alt="License">
  <a href="https://discord.gg/H4XZwsYzY4">
    <img src="https://img.shields.io/discord/1212548097624903681.svg?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

<div align="center">
  <a href="https://docs.tracecat.com">
    <img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xMWgtNmEzIDMgMCAwIDAtMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white" alt="Documentation">
  </a>
  <a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates">
    <img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white" alt="Templates Library">
  </a>
</div>

## Tracecat: Automate Security & IT Workflows with Ease

Tracecat is an open-source workflow automation platform designed to streamline security and IT response engineering, providing a centralized hub for your automation needs.  [Learn more on GitHub](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-based Templates:** Define integrations and workflows using simple, human-readable YAML, making automation accessible.
*   **No-Code UI:** Easily build and manage workflows with a user-friendly, no-code interface.
*   **Built-in Lookup Tables:** Simplify data management and enrich your workflows with built-in lookup tables.
*   **Case Management:**  Manage and track incidents efficiently with integrated case management capabilities.
*   **Scalable Orchestration:** Powered by Temporal for robust and scalable workflow execution.
*   **Open Cyber Security Schema (OCSF) Compatibility:**  Templates align with OCSF for consistent data formatting.
*   **Extensive Template Library:** Access a growing library of pre-built templates for common security and IT tasks.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Find full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Find full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Have questions, feedback, or integration ideas?  Join the Tracecat community on [Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

Tracecat Registry offers a collection of pre-built integration and response-as-code templates. These actions are organized using Tracecat's ontology of common capabilities, aligning with OCSF where possible.

**Examples:**

Visit our documentation on Tracecat Registry for use cases and ideas.  Explore existing open-source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is available under the AGPL-3.0 license, excluding the `ee` directory, which contains paid enterprise features. The Enterprise Edition offers advanced features requiring specific investments.  You can enable the Enterprise Edition directly in the platform's settings.

*For information on Tracecat's Enterprise self-hosted or managed Cloud offering, visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

SSO, audit logs, and IaC deployments (Terraform, Kubernetes/Helm) are free and available.  We are developing a comprehensive threat model, security features, and hardening recommendations. For immediate security inquiries, please reach out on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including `tracecat` in the subject line.

## Contributors

Thank you to all our contributors for your code, integrations, and support!  Open source thrives because of you. ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" alt="Contributors">
</a>

<br>
<br>

<div align="center">
  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>
</div>