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

## Tracecat: Automate Security and IT Workflows with Open Source

Tracecat is an open-source workflow automation platform designed to streamline security and IT operations. Built for speed and reliability, Tracecat empowers engineers to automate complex tasks with ease.  [Check out the original repo](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-Based Templates:** Define workflows and integrations using simple, human-readable YAML templates.
*   **No-Code UI:**  Build and manage workflows visually with Tracecat's intuitive user interface.
*   **Built-in Lookup Tables & Case Management:**  Easily store and access data for enhanced context and collaboration.
*   **Scalable Architecture:**  Leverages Temporal for robust and scalable workflow orchestration.
*   **Open Cyber Security Schema (OCSF) Integration:** Standardizes inputs and outputs for seamless interoperability.

![Tracecat workflow](/img/workflow.png)

## Getting Started

**Important:** Tracecat is under active development. Please review the [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Locally with Docker Compose

Deploy a local Tracecat instance quickly using Docker Compose.  Full instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Deploy on AWS Fargate (Advanced)

For production environments, deploy Tracecat on AWS Fargate using Terraform.  Detailed instructions are provided [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Kubernetes Deployment (Coming Soon)

Stay tuned for instructions on deploying Tracecat on Kubernetes.

## Community

Join the Tracecat community! Share ideas, ask questions, and collaborate with other users on [Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry: Integration Templates

![Tracecat Action template](img/action-template.svg)

Tracecat Registry provides a growing library of pre-built integration templates, enabling rapid deployment of security and IT automation solutions. Actions are categorized using Tracecat's ontology of common capabilities, and inputs are standardized with OCSF for better data consistency.

**Key Resources:**

*   **Use Cases & Examples:**  Explore practical applications in our documentation.
*   **Open Source Templates:**  Browse and contribute to the open-source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

Tracecat is primarily licensed under AGPL-3.0, with the exception of the `ee` directory, which contains features for the Tracecat Enterprise license.

*   **Open Source:** Core platform functionality, including essential features like SSO, audit logs, and IaC deployments, are available under the AGPL-3.0 license.
*   **Enterprise Edition:** Offers advanced features and capabilities for enhanced security and performance. For information on the Enterprise Edition, visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security.  Free features will include SSO, audit logs, and IaaC deployments.  We're working on a comprehensive threat model and security best practices documentation.

*   **Report Security Issues:**  Report vulnerabilities to [security@tracecat.com](mailto:founders+security@tracecat.com) with "tracecat" in the subject line.

## Contributors

Thank you to all contributors! Your contributions are essential to making Tracecat a success.

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>