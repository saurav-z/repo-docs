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

<a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xMWgtNmEzIDMgMCAwIDAtMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
<a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>

</div>

## Tracecat: Automate Security and IT Workflows with Open Source

Tracecat is a modern, open-source automation platform empowering security and IT engineers to streamline their workflows.  [Learn more at the original repo](https://github.com/TracecatHQ/tracecat).

### Key Features

*   **YAML-Based Templates:** Easily define integrations using simple, human-readable YAML templates.
*   **No-Code UI:**  Build and manage workflows with a user-friendly, no-code interface.
*   **Built-in Lookup Tables & Case Management:** Organize and manage your data efficiently.
*   **Reliable Orchestration:** Powered by Temporal for scalable and reliable execution.
*   **Open Cyber Security Schema (OCSF) Integration:** Supports the OCSF standard for streamlined data normalization.
*   **Tracecat Registry:** Access a library of pre-built templates for common security and IT tasks.
*   **Flexible Deployment Options:**  Deploy locally with Docker Compose, on AWS Fargate, and soon on Kubernetes.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform.  Find comprehensive instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry provides a collection of pre-built integration and response-as-code templates. Response actions are organized using Tracecat's ontology of common capabilities, and template inputs are normalized to fit the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) where possible.

**Examples**

Explore the Tracecat Registry documentation for use cases and ideas.
Check out existing open source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join the Tracecat community on [Discord](https://discord.gg/H4XZwsYzY4) to ask questions, provide feedback, and share integration ideas.

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, excluding the `ee` directory which contains features exclusive to Tracecat Enterprise.  The Enterprise Edition provides additional features developed through significant R&D investment. For information on self-hosted or managed Cloud offerings, please visit [the website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat offers free SSO, audit logs, and IaC deployment options (Terraform, Kubernetes/Helm).  A detailed threat model and security recommendations are under development.  For immediate security questions, reach out on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including "tracecat" in the subject line.

## Contributors

We appreciate all our contributors! ❤️

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

*   **SEO Optimization:**  Uses keywords like "automation platform," "security," "IT workflows," "open source," and "response engineering" throughout the README. Includes a concise one-sentence hook.
*   **Clear Structure:** Uses headings (H2) to organize the content for readability and searchability.
*   **Bulleted Key Features:**  Highlights the core benefits and functionalities in an easy-to-scan list, appealing to users quickly.
*   **Community Links:** Directs users to the Discord community and provides a clear call to action.
*   **Concise Language:**  Streamlines the descriptions for better clarity and impact.
*   **Emphasis on Benefits:** Focuses on *what* Tracecat does for users rather than just *what* it is.
*   **Clear Calls to Action:**  Encourages users to engage with the community, explore the documentation, and potentially consider the Enterprise offering.
*   **Link Back to Original Repo:** Maintains the link to the original repo at the top.
*   **Corrected Formatting and Typos:** minor improvements.
*   **OCSF and Registry emphasis:** Included details about OCSF support and the registry to highlight key features.
*   **Developer friendly:** Kept the essential "Getting Started" section, but also added more about AWS Fargate which is attractive for more technical users.