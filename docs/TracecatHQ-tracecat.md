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

# Tracecat: Automate Security & IT Workflows with Open Source Power

Tracecat is an open-source workflow automation platform, empowering security and IT teams to streamline incident response and automate tasks with ease.  ([View on GitHub](https://github.com/TracecatHQ/tracecat))

## Key Features

*   **YAML-Based Templates:**  Define workflows with simple, human-readable YAML files.
*   **No-Code UI:**  Build and manage workflows through an intuitive, user-friendly interface.
*   **Built-in Lookup Tables & Case Management:**  Organize and track incidents efficiently.
*   **Scalable & Reliable:** Built on Temporal for robust performance.
*   **Open Cyber Security Schema (OCSF) Integration:**  Ensures data consistency and interoperability.

## Getting Started

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Full instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (For Advanced Users)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. View instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry: Pre-built Integrations

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry offers a library of pre-built integration templates. These templates use a standardized ontology of common capabilities, making automation consistent.

**Key Benefits:**

*   **Simplified Integration:**  Leverage pre-built templates for common security and IT tasks.
*   **Standardized Capabilities:**  Actions are organized using Tracecat's ontology.
*   **OCSF Compatibility:**  Template inputs are normalized with the Open Cyber Security Schema where possible.

**Explore the Registry:**
Visit our documentation for use cases: [Tracecat Registry documentation](https://docs.tracecat.com)
Or explore existing open source templates: [Tracecat Registry Templates](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join the Tracecat community! Get support, share ideas, and collaborate on new integrations on our [Discord Server](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, excluding the `ee` directory, which contains features requiring a Tracecat Enterprise license.

*   **Open Source:** Offers robust core functionality for building automated workflows.
*   **Enterprise Edition:** Provides advanced features to enhance the platform.

If you're interested in Tracecat's Enterprise self-hosted or managed Cloud offering, check out [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security with features like SSO, audit logs, and infrastructure-as-code (IaC) deployment options (Terraform, Kubernetes/Helm).

For detailed information on Tracecat's threat model, security features, and hardening recommendations, contact us on [Discord](https://discord.gg/H4XZwsYzY4).

Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) and include `tracecat` in the subject line.

## Contributors

Thank you to all our contributors who make Tracecat possible!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>
```
Key changes and SEO optimization:

*   **Clear, concise title and description:**  The hook is right at the top.
*   **Keywords:**  Includes terms like "security automation," "IT automation," "incident response," "open source," and "workflow automation" for search optimization.
*   **Bulleted Features:**  Easy to scan and highlight key benefits.
*   **Strong Calls to Action:** Directs users to the documentation, Discord, and GitHub.
*   **Organized Headings:** Improves readability and SEO.
*   **Contextual Links:**  Uses descriptive link text for SEO benefits (e.g., "View on GitHub", "Tracecat Registry documentation").
*   **Removed Redundancy:** Streamlined the text to focus on the core value proposition.
*   **Community emphasis**: Expanded on how to engage with the community.
*   **Concise security details**: Focused on key points.