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

## Tracecat: Automate Security and IT Workflows with Open Source Power

Tracecat is a modern, open-source platform empowering security and IT engineers to automate their workflows using simple YAML-based templates and a no-code UI. **[Check out the Tracecat repository](https://github.com/TracecatHQ/tracecat)!**

### Key Features:

*   **Workflow Automation:** Design and execute complex workflows using YAML templates.
*   **No-Code UI:** Easily manage and monitor your workflows with a user-friendly interface.
*   **Built-in Lookup Tables & Case Management:** Simplify data management and incident response.
*   **Scalable & Reliable:** Built on Temporal for robust performance and scalability.
*   **Open Source:** Leverage the power of open-source for security and IT automation.
*   **Tracecat Registry:** Utilize the collection of integration and response-as-code templates.
*   **Open Cyber Security Schema (OCSF) Integration:** Inputs are normalized where possible.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. View full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** deploy a production-ready Tracecat stack on AWS Fargate using Terraform. View full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community to ask questions, provide feedback, and share integration ideas.

*   [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4)

## Tracecat Registry: Response-as-Code Templates

![Tracecat Action template](img/action-template.svg)

Tracecat Registry provides a curated collection of integration and response-as-code templates. Response actions are organized using Tracecat's ontology of common capabilities (e.g., `list_alerts`, `list_cases`, `list_users`).

*   [Tracecat Registry Documentation](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose)
*   [Open Source Templates](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates)

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, with the `ee` directory containing paid enterprise features.

*   Learn more about [Tracecat's Enterprise self-hosted or managed Cloud offering](https://tracecat.com) or [book a meeting with us](https://cal.com/team/tracecat).

## Security

Tracecat is committed to security.

*   SSO, audit logs, and Infrastructure as Code (IaaC) deployments (Terraform, Kubernetes/Helm) are always free and available.
*   Comprehensive information on Tracecat's threat model, security features, and hardening recommendations is coming soon.
*   For immediate security questions, reach out on [Discord](https://discord.gg/H4XZwsYzY4).
*   Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## Contributors

Thank you to all our contributors!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>
```
Key changes and improvements:

*   **SEO Optimization:**  Added a concise, keyword-rich headline, a descriptive summary and added keywords like "security," "IT automation," "workflows," "open source," "response-as-code."
*   **One-Sentence Hook:** A compelling first sentence.
*   **Clear Headings:** Used headings for better readability and SEO.
*   **Bulleted Key Features:**  Made the key features stand out.
*   **Conciseness:**  Removed redundant information.
*   **Community Links:**  Made community links more prominent.
*   **Call to Action:**  Encouraged users to check out the repo.
*   **Emphasis on Open Source:**  Highlighted the open-source nature of the project.
*   **Improved Formatting:** Improved formatting for readability.
*   **Links:**  Added links to important pages.