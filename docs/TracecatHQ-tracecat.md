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

## Tracecat: Automate Security & IT Workflows with Ease

Tracecat is an open-source workflow automation platform designed for security and IT engineers, offering powerful tools to streamline your incident response and IT operations. **Check out the original repo [here](https://github.com/TracecatHQ/tracecat).**

**Key Features:**

*   **YAML-Based Templates:** Easily define integrations and workflows using simple, human-readable YAML.
*   **No-Code UI:** Build and manage workflows with an intuitive, no-code user interface.
*   **Built-in Lookup Tables & Case Management:** Simplify data enrichment and incident tracking.
*   **Scalable & Reliable Orchestration:** Powered by Temporal for robust and scalable workflow execution.
*   **Tracecat Registry:** Access a growing library of pre-built integration templates.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  [Get full instructions here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Deploy on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. [View full instructions here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Kubernetes Deployment (Coming Soon)

Stay tuned for Kubernetes deployment instructions.

## Community

Join the Tracecat community! Share your questions, feedback, and integration ideas in the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

Tracecat Registry provides a centralized repository of integration and response-as-code templates. Response actions are structured using Tracecat's ontology of capabilities (e.g., `list_alerts`, `list_cases`, `list_users). Template inputs are normalized to the Open Cyber Security Schema (OCSF) where possible.

**Explore:**

*   Visit the Tracecat Registry documentation for use cases and ideas.
*   Browse the open-source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, with the exception of the `ee` directory which contains paid enterprise features. The Enterprise Edition offers advanced features developed through dedicated R&D. You can enable the Enterprise Edition within the platform settings.

*For self-hosted or managed Cloud offering details, explore the [Tracecat website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

Tracecat prioritizes security, offering features like SSO, audit logs, and IaaC deployments (Terraform, Kubernetes / Helm) available in the open source version. A comprehensive threat model and security hardening recommendations are in development. For immediate security inquiries, contact us on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## Contributors

A big thank you to our contributors for their code, integrations, and support!  Open source thrives because of you. ❤️

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

*   **SEO-Optimized Title & Hook:**  The first sentence now acts as a strong hook, immediately stating what Tracecat is and its value proposition, and linking back to the repo.
*   **Clear Headings:** Uses clear and concise headings for easy navigation.
*   **Bulleted Key Features:**  Uses bullet points to highlight key features, making it easy to scan and understand the platform's capabilities.
*   **Keyword Optimization:** Incorporated keywords like "security," "IT," "workflow automation," "incident response," "open source," and "templates" to improve search visibility.
*   **Concise Descriptions:**  Rewrote descriptions to be more concise and action-oriented.
*   **Call to Action:** Includes calls to action, like "Join the Tracecat community!"
*   **Community Links:** Provides easy access to the Discord community.
*   **Documentation Links:**  Includes links to relevant documentation.
*   **Licensing Information:**  Maintains the original licensing information.
*   **Contributor Section:**  Keeps the contributor section to give credit to contributors.
*   **Overall Readability:** Improved overall readability and organization.