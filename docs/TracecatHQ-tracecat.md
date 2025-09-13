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

# Tracecat: Automate Security and IT Workflows with Ease

**Tracecat** is an open-source automation platform designed to streamline security and IT response engineering. Visit the [Tracecat GitHub repository](https://github.com/TracecatHQ/tracecat).

## Key Features

*   **YAML-Based Templates:** Define integrations and workflows using simple, human-readable YAML templates.
*   **No-Code UI:** Leverage a user-friendly interface for workflow creation and management.
*   **Built-in Lookup Tables & Case Management:** Simplify data handling and incident tracking within the platform.
*   **Scalable & Reliable Orchestration:** Powered by Temporal for robust and scalable workflow execution.
*   **Tracecat Registry:** A library of response actions, organized by common capabilities.
*   **Open Cyber Security Schema (OCSF) Integration:**  Normalize data inputs for seamless compatibility with industry standards.

![Tracecat workflow](/img/workflow.png)

## Getting Started

### Run Tracecat Locally

Quickly deploy a local Tracecat instance using Docker Compose. For detailed instructions, refer to the [Docker Compose deployment guide](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform.  Instructions can be found [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

*Coming soon.*

## Community & Support

Join the Tracecat community for questions, feedback, and new integration ideas: [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry: Templates for Automation

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry is a curated collection of integration and response-as-code templates. These templates streamline common tasks and enhance automation capabilities. Explore pre-built integrations and contribute your own: [Tracecat Registry Repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

**Examples**

*   Review the documentation on Tracecat Registry for use cases and ideas.

## Open Source vs. Enterprise

Tracecat is available under the AGPL-3.0 license, excluding the `ee` directory containing enterprise features. The Enterprise Edition offers advanced functionalities and requires a Tracecat Enterprise license.

*For information on Tracecat's Enterprise offering, including self-hosted and managed cloud options, visit [Tracecat's website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

Tracecat prioritizes security with features like SSO, audit logs, and Infrastructure as Code (IaaC) deployments (Terraform, Kubernetes / Helm).

Report security vulnerabilities to: [security@tracecat.com](mailto:founders+security@tracecat.com) (Subject: `tracecat`).

## Contributors

A special thanks to our contributors. Open source is a community effort, and we appreciate everyone who contributes code, integrations, and support. ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>