<div align="center">
  <img src="img/banner.svg" alt="The workflow automation platform for security and IT response engineering.">
</div>

<br>

<div align="center">

![Commits](https://img.shields.io/github/commit-activity/m/TracecatHQ/tracecat?style=for-the-badge&logo=github)
![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge&logo=agpl)
[![Discord](https://img.shields.io/discord/1212548097624903681.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/H4XZwsYzY4)

</div>

<div align="center">

<a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xMWgtNmEzIDMgMCAwIDAtMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
<a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>

</div>

## Tracecat: Automate Security and IT Workflows with Open Source Power

Tracecat is an open-source automation platform that empowers security and IT teams to streamline incident response and operational workflows with ease. ([View on GitHub](https://github.com/TracecatHQ/tracecat))

**Key Features:**

*   **YAML-Based Templates:** Define integrations and workflows with simple, human-readable YAML.
*   **No-Code UI:**  Easily build and manage workflows through an intuitive user interface.
*   **Built-in Lookup Tables & Case Management:**  Efficiently manage data and track incidents.
*   **Scalable & Reliable:** Orchestrated using Temporal for robust performance.
*   **Open Cyber Security Schema (OCSF) Compliance:**  Templates leverage OCSF for standardized data inputs.
*   **Extensible Template Library:** Access a growing collection of pre-built integrations and response actions in the [Tracecat Registry](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Find detailed instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (Advanced)

For production environments, deploy Tracecat on AWS Fargate using Terraform.  Full instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon!

## Community

Join the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4) to ask questions, provide feedback, and share integration ideas.

## Tracecat Registry: Integration & Response Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry provides a curated collection of integration and response-as-code templates, organized by common capabilities, and following the Open Cyber Security Schema (OCSF).

**Examples:**

*   `list_alerts`
*   `list_cases`
*   `list_users`

Explore the Tracecat Registry for use cases and examples in our documentation.  You can also check out existing open source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is available under the AGPL-3.0 license, with the exception of the `ee` directory, which contains proprietary enterprise features requiring a Tracecat Enterprise license. The Enterprise Edition offers advanced capabilities requiring specific investments in research and development.

You can enable the Enterprise Edition directly within the platform settings.

*For information on Tracecat's Enterprise self-hosted or managed Cloud offerings, visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

SSO, audit logs, and IaC deployments (Terraform, Kubernetes/Helm) are always free and available.  We are developing a comprehensive document on Tracecat's threat model, security features, and hardening recommendations.  For immediate security questions, please reach out on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## Contributors

Thank you to all our contributors for your code, integrations, and support.  Open source thrives because of you! ❤️

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

*   **Clear Headline:**  Uses "Tracecat: Automate Security and IT Workflows" - this is a strong keyword phrase.
*   **One-Sentence Hook:** Immediately grabs attention and clearly defines the product.
*   **Keyword Optimization:** Repeatedly uses relevant keywords like "automation," "security," "IT," "workflows," "incident response," and "open source".
*   **Concise Key Features:**  Uses bullet points for easy readability and quick understanding.
*   **Strong Calls to Action:**  Links to the GitHub repo, documentation, Discord, and website.
*   **Clear Structure:** Uses headings and subheadings to organize information logically and for better SEO.
*   **Emphasis on Benefits:** Highlights the value proposition of Tracecat, such as ease of use, scalability, and compliance.
*   **Contextual Links:** Links to the relevant documentation and resources throughout.
*   **Contributor Section:**  Includes the contributors' section to foster a community.
*   **Security Section:** Added a dedicated security section to highlight the platform's safety and encourage users to report any issues.
*   **Updated and improved the tone** by fixing grammatical errors and making the writing more concise.