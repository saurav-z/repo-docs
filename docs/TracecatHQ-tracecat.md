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

## Tracecat: Automate Security and IT Workflows with Open Source Power

Tracecat is an open-source workflow automation platform, designed to streamline security and IT response engineering. <a href="https://github.com/TracecatHQ/tracecat">Explore the code on GitHub.</a>

**Key Features:**

*   **YAML-Based Templates:** Easily define integrations with simple, human-readable YAML templates.
*   **No-Code UI for Workflows:** Build and manage complex workflows with an intuitive no-code user interface.
*   **Built-in Lookup Tables & Case Management:** Simplify data management and incident handling.
*   **Scalable & Reliable Orchestration:** Powered by Temporal for robust and scalable workflow execution.
*   **Template Registry:** A growing library of pre-built integrations and response actions.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Instructions can be found [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community and connect with other users. Share your questions, feedback, and integration ideas on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry: Integration and Response-as-Code Templates

![Tracecat Action template](img/action-template.svg)

Tracecat Registry provides a centralized collection of integration and response-as-code templates, organized by common capabilities. Templates use the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) ontology, ensuring data consistency where possible.

**Examples:**

*   Refer to our documentation on Tracecat Registry for use cases and ideas.
*   Explore existing open source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, except for the `ee` directory, which contains paid enterprise features requiring a Tracecat Enterprise license. The Enterprise Edition offers advanced features that require specific research and development investments.

*For information on Tracecat's Enterprise self-hosted or managed Cloud offering, please visit our website ([https://tracecat.com](https://tracecat.com)) or [book a meeting](https://cal.com/team/tracecat).*

## Security

Tracecat offers robust security features, including SSO and audit logs, and supports IaC deployments (Terraform, Kubernetes/Helm). A comprehensive list of threat models, security features, and hardening recommendations is being compiled.

For immediate security questions, please reach out on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## Contributors

Thank you to all our contributors for their code, integrations, and support.

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

*   **SEO-Optimized Title & Hook:**  The title includes relevant keywords ("workflow automation," "security," "IT response"). The hook sentence directly addresses the user's need.
*   **Clear Headings:**  Improved structure with clear, concise headings for each section.
*   **Bulleted Key Features:** Highlights the core benefits in an easy-to-read format.
*   **Concise Language:** Avoids overly verbose descriptions.
*   **Call to Action:** Encourages users to visit the docs, join the Discord, and explore the templates.
*   **Contextual Links:**  Links are included inline to provide context and support.
*   **Contributor Section:**  Kept and emphasized to show appreciation.
*   **License Information:** Remains and is clearly displayed.
*   **GitHub Link Emphasis:** The "Explore the code on GitHub" link is now in the introductory sentence.
*   **Keywords Throughout:** Keywords like "automation," "security," and "IT" are used naturally throughout the text.
*   **OCSF Mention Improved:**  Made clearer why OCSF is used.
*   **Removed Redundancy:** Consolidated some of the information to keep the README concise.
*   **More Action-Oriented Language:** Uses phrases like "Explore," "Join," and "Report" to encourage user interaction.
*   **Updated Discord Link:** Maintained current links and badge.