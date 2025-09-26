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

## Tracecat: Automate Security and IT Workflows with Open Source

**Tracecat is a modern, open-source automation platform designed to empower security and IT engineers to streamline and orchestrate their critical workflows.**

[View the Tracecat repository on GitHub](https://github.com/TracecatHQ/tracecat)

**Key Features:**

*   **YAML-Based Templates:** Easily define integrations and workflows using simple, human-readable YAML templates.
*   **No-Code UI:** Leverage a user-friendly interface to build and manage workflows without writing code.
*   **Built-in Lookup Tables & Case Management:** Simplify data handling and incident tracking with integrated lookup tables and case management features.
*   **Scalable Orchestration:** Powered by Temporal for robust, reliable, and scalable workflow execution.
*   **Template Registry:** Access a growing library of pre-built templates for common security and IT tasks.
*   **Open Cyber Security Schema (OCSF) Support:** Template inputs are normalized to align with OCSF for improved interoperability.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Find the full guide [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community to ask questions, provide feedback, and share integration ideas. Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

Tracecat Registry is a comprehensive collection of integration and response-as-code templates. Response actions are organized using Tracecat's ontology of common capabilities (e.g. `list_alerts`, `list_cases`, `list_users`).

**Examples**

Explore use cases and ideas in our documentation on Tracecat Registry.
Discover existing open-source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs Enterprise

This repository is licensed under AGPL-3.0, excluding the `ee` directory, which contains paid enterprise features requiring a Tracecat Enterprise license.

The Enterprise Edition provides advanced features that require dedicated research and development investments.

You can enable the Enterprise Edition directly in the platform's settings.

*For information about Tracecat's Enterprise self-hosted or managed Cloud offerings, visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

SSO, audit logs, and IaaC deployments (Terraform, Kubernetes / Helm) are freely available.  We are developing a comprehensive security overview of Tracecat's threat model, security features, and hardening recommendations. Reach out on [Discord](https://discord.gg/H4XZwsYzY4) for immediate security inquiries.

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with "tracecat" in the subject line.

## Contributors

Thank you to our amazing contributors for their code, integrations, and support. Open source thrives because of you. ❤️

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

*   **SEO-Optimized Title & Hook:**  A compelling one-sentence hook at the beginning grabs attention. The title includes relevant keywords ("workflow automation," "security," "IT").
*   **Clear Headings:**  Organized with clear headings for easy navigation and readability.
*   **Bulleted Key Features:** Uses bullet points to highlight the most important aspects of the platform, improving scannability.
*   **Concise Descriptions:**  Keeps descriptions short and to the point, using active voice.
*   **Call to Action:** The "Getting Started" section and the "Community" section offer clear calls to action.
*   **Keyword Optimization:**  Repeatedly mentions relevant keywords like "automation," "security," "IT," "workflows," and "templates."
*   **Structure and Formatting:** Uses markdown effectively for headings, bold text, and lists.
*   **Improved Emphasis:** Uses the `[!IMPORTANT]` block to visually highlight important warnings.
*   **Link Back to the Repo:** Included the link back to the original repository.
*   **Contributors Section:** Keeps the contributors section as is.
*   **Concise and Informative:** Provides a good overview without being overly verbose.
*   **Documentation Links:** Provides direct links to relevant documentation pages.