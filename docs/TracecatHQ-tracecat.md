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

## Tracecat: Automate Security & IT Workflows with Open Source Power

Tracecat is an open-source platform that empowers security and IT teams to automate complex workflows with ease.  [View the original repository](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-Based Templates:**  Define integrations and workflows using simple, easy-to-understand YAML files.
*   **No-Code UI:**  Build and manage workflows visually with our intuitive user interface.
*   **Built-in Lookup Tables:** Streamline data management and enhance workflow logic with integrated lookup tables.
*   **Case Management:**  Organize and track incidents efficiently with built-in case management features.
*   **Reliable Orchestration:** Powered by Temporal for scalability and dependable performance.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Full instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. See instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community for questions, feedback, and new integration ideas!  Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry: Integration Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry offers a library of pre-built integration and response-as-code templates to kickstart your automation.

*   **Action Ontology:** Actions are organized by common capabilities (e.g., `list_alerts`, `list_cases`).
*   **OCSF Compliance:** Template inputs are normalized to fit the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) where applicable.

**Examples & Templates**

Explore use cases and ideas in our documentation.
Browse open-source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, with the exception of the `ee` directory, which contains paid enterprise features.  The Enterprise Edition offers advanced capabilities requiring dedicated development.  Enable the Enterprise Edition directly in the platform's settings.

For information on Tracecat's Enterprise self-hosted or managed Cloud offerings, visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

SSO, audit logs, and Infrastructure-as-Code deployments (Terraform, Kubernetes/Helm) are free and always available.  We are developing a comprehensive security guide, including threat models and hardening recommendations. For immediate security questions, reach out on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject.

## Contributors

Thank you to our amazing contributors for their code, integrations, and support.

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>
```
Key improvements and SEO considerations:

*   **Clear Headline:**  The main headline immediately grabs attention.
*   **Concise Hook:** The opening sentence is compelling and describes what the project *does*.
*   **Keyword Integration:**  Keywords like "security automation", "IT workflow", "open source" are used.
*   **Bulleted Features:** Uses bullet points to highlight key selling points for easy skimming.
*   **Strong Call to Action:**  Links to the documentation and community are prominent.
*   **SEO-Friendly Formatting:**  Uses headings, bold text, and other formatting elements for readability.
*   **Relevant Links:**  Provides easy access to important resources.
*   **Community Engagement:** Directs users to the Discord.
*   **Contributor Image:** Displays the contributor list, increasing engagement and helping SEO.
*   **License Information:**  Keeps the important license information and is easily accessible.
*   **Rephrased and Improved Content:** Clarified existing content to be more concise and easier to digest.
*   **Removed Redundancy:** Consolidated some of the information.