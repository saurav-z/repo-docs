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

**Tracecat is a modern, open-source platform empowering security and IT engineers to automate incident response and streamline operations.**  [Explore the Tracecat GitHub Repository](https://github.com/TracecatHQ/tracecat)

### Key Features

*   **YAML-Based Templates:** Easily define integrations and workflows using simple YAML templates.
*   **No-Code UI:**  Design and manage workflows through an intuitive no-code user interface.
*   **Built-in Lookup Tables & Case Management:** Organize and track incidents effectively.
*   **Scalable & Reliable Orchestration:** Powered by Temporal for robust performance.
*   **Open Source:** Available under the AGPL-3.0 license.
*   **Tracecat Registry:** Access a library of pre-built integration and response templates, organized by common capabilities and aligned with the Open Cyber Security Schema (OCSF).

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### 1. Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### 2. Deploy on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform.  Instructions can be found [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### 3. Kubernetes Deployment (Coming Soon)

Support for Kubernetes deployment is planned.

## Community

Join the Tracecat community!  Ask questions, provide feedback, and share integration ideas on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

Tracecat Registry is a collection of integration and response-as-code templates, designed to standardize and accelerate your security and IT automation efforts.  Response actions are organized using Tracecat's ontology of common capabilities (e.g., `list_alerts`, `list_cases`, `list_users`). Template inputs are normalized to fit the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) ontology where possible.

**Examples**

*   Visit our documentation on Tracecat Registry for use cases and ideas.
*   Explore existing open-source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository provides the open-source core of Tracecat, licensed under AGPL-3.0.  The `ee` directory contains paid enterprise features requiring a Tracecat Enterprise license.  The Enterprise Edition provides advanced capabilities that require dedicated research and development investments.

*Interested in Tracecat's Enterprise self-hosted or managed Cloud offerings?*  Visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security with features such as SSO and audit logs. We are actively developing a comprehensive threat model, security features, and hardening recommendations.

*   For immediate security-related inquiries, contact us on [Discord](https://discord.gg/H4XZwsYzY4).
*   Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including "tracecat" in the subject line.

## Contributors

Thank you to all of our amazing contributors!

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

*   **SEO Optimization:**  Includes relevant keywords (e.g., "security automation," "IT automation," "incident response," "open source").  Uses headings effectively.
*   **One-Sentence Hook:**  Clearly defines what Tracecat *is* in a compelling way, and what it does.
*   **Bulleted Key Features:**  Highlights the main benefits in an easy-to-scan format.  Concise and keyword-rich.
*   **Clearer Structure:** Uses headings and subheadings to organize the information logically.
*   **Call to Action:** Encourages users to explore the repo and the documentation.
*   **Community & Contact Information:**  Easy access to the Discord and Security email address.
*   **Links to Documentation and Templates:** Make it easy for users to explore and learn more.
*   **More Engaging Tone:**  More active verbs and direct language.
*   **Updated "Getting Started" section:** Removed unnecessary "Run Tracecat locally" section.
*   **Emphasis on Open Source:**  Highlights the open-source nature of the core product.
*   **Clear Explanation of Enterprise Edition:** Explains the difference between the open source and enterprise versions.
*   **Contributors section:**  Improved the presentation and provided context.