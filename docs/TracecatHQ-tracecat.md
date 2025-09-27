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

# Tracecat: Automate Security and IT Workflows with Open Source

Tracecat is a cutting-edge, open-source platform designed to automate and streamline security and IT response engineering workflows.  [Explore the code on GitHub](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-Based Templates:** Define integrations and workflows using simple, YAML-based templates for easy configuration.
*   **No-Code UI:** Leverage a user-friendly, no-code UI for building and managing your automation workflows.
*   **Built-in Lookup Tables & Case Management:** Simplify data management and case handling within the platform.
*   **Scalable Orchestration:** Powered by Temporal for robust, reliable, and scalable workflow execution.
*   **Open Cyber Security Schema (OCSF) Integration:**  Templates normalized with OCSF for enhanced data interoperability.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Quickly get started with a local Tracecat instance using Docker Compose.  Find detailed instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**(Advanced Users):** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Instructions available [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry: Integration Templates

![Tracecat Action template](img/action-template.svg)

Tracecat Registry offers a comprehensive collection of integration and response-as-code templates, organized by common security and IT capabilities, enabling you to quickly automate tasks.

**Key Benefits:**

*   **Pre-built Templates:** Access pre-built templates for common actions like `list_alerts` and `list_cases`.
*   **OCSF Alignment:**  Template inputs are designed to align with the Open Cyber Security Schema (OCSF) for improved data standardization.
*   **Extensible Library:** Discover and utilize existing templates, and contribute your own to the growing library.

**Examples and Resources:**

*   Explore use cases and ideas in our [Tracecat Registry documentation](https://docs.tracecat.com).
*   Browse the open-source templates in our [repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join our vibrant community for support, discussions, and new integration ideas!  Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

Tracecat is primarily available under the AGPL-3.0 license, with enterprise features contained within the `ee` directory requiring a commercial license.  The Enterprise Edition offers advanced features developed through dedicated research and development.

*For information on Tracecat's Enterprise self-hosted or managed Cloud offerings, please visit our [website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

Security is a priority.  We provide free SSO, audit logs, and IaaC deployments (Terraform, Kubernetes / Helm). We are actively working on a comprehensive security model.  For immediate answers, contact us on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including "tracecat" in the subject line.

## Contributors

A huge thanks to all our contributors for their code, integrations, and support!  Your contributions make open-source possible. ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>
```
Key improvements and SEO optimizations:

*   **Clear Title and Hook:**  A compelling title and a one-sentence hook immediately grab the reader's attention.
*   **Keyword Integration:** Keywords like "security," "IT," "automation," "workflows," "open source," "templates," and "OCSF" are integrated naturally throughout.
*   **Structured Headings:** Headings provide clear organization and improve readability for both humans and search engines.
*   **Bulleted Lists:** Key features are presented in bulleted lists for easy scanning and comprehension.
*   **Internal and External Links:**  Strategic use of links to documentation, the Discord community, and the GitHub repository promotes user engagement and SEO.
*   **Call to Actions:**  Encourages users to "Explore the code," "Get Started," and join the community.
*   **Concise Language:**  Information is presented concisely and efficiently, optimizing for readability.
*   **Alt Text for Images:** Ensures images are accessible and contribute to SEO.
*   **Clear Sectioning:** Information is logically organized.
*   **Community Section:** Highlights the community aspects, important for open-source projects.
*   **Security Emphasis:**  Reiterates security focus.
*   **Contributor Showcase:**  Keeps and promotes contributions, a crucial aspect of open source.