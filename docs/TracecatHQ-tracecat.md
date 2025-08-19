<div align="center">
  <img src="img/banner.svg" alt="Tracecat: The workflow automation platform for security and IT response engineering.">
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

## Tracecat: Automate Security and IT Workflows with Ease

Tracecat is an open-source workflow automation platform designed to streamline security and IT operations, providing a user-friendly environment for engineers to build and manage automated responses. **[Explore the Tracecat GitHub Repository](https://github.com/TracecatHQ/tracecat)**

**Key Features:**

*   **YAML-Based Templates:** Simplify integrations with easy-to-understand YAML templates.
*   **No-Code UI for Workflows:**  Build and manage workflows visually without writing code.
*   **Built-in Lookup Tables and Case Management:** Organize data and track incidents efficiently.
*   **Scalable Orchestration with Temporal:** Ensures reliability and scalability for complex automation tasks.
*   **Open Cyber Security Schema (OCSF) Integration:** Ensures standardized data inputs.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Get started quickly with Docker Compose. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Deploy on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. See full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Kubernetes Deployment (Coming Soon)

Stay tuned for Kubernetes deployment instructions.

## Community

Join the Tracecat community on [Discord](https://discord.gg/H4XZwsYzY4) for questions, feedback, and new integration ideas!

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

Tracecat Registry is your source for pre-built integration and response-as-code templates. Response actions are organized using Tracecat's ontology of common capabilities (e.g. `list_alerts`, `list_cases`, `list_users`). Template inputs are normalized to fit the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) ontology where possible.

**Examples**

Visit our documentation on Tracecat Registry for use cases and ideas.
Or check out existing open source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

The core Tracecat platform is available under the AGPL-3.0 license, with the exception of the `ee` directory, which contains paid enterprise features requiring a Tracecat Enterprise license.

*For information on Tracecat's Enterprise self-hosted and managed Cloud offering, please visit [our website](https://tracecat.com) or [book a meeting with us](https://cal.com/team/tracecat).*

## Security

Tracecat prioritizes security. We offer SSO, audit logs, and IaC deployments (Terraform, Kubernetes/Helm) for free. We're developing a comprehensive threat model, security features, and hardening recommendations. For immediate answers, reach out to us on [Discord](https://discord.gg/H4XZwsYzY4).

Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including `tracecat` in the subject line.

## Contributors

Huge thanks to our amazing contributors for their code, integrations, and support.  Open source thrives because of you! ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">
  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>
</div>