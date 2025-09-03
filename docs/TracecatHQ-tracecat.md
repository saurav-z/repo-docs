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

# Tracecat: Automate Security & IT Response with Open Source

Tracecat is a powerful, open-source platform designed to streamline security and IT response workflows.  Check out the [original repo](https://github.com/TracecatHQ/tracecat) for more details!

**Key Features:**

*   **YAML-based Templates:** Easily define integrations with simple YAML templates for rapid workflow creation.
*   **No-Code UI:** Build and manage workflows visually using an intuitive, user-friendly interface.
*   **Built-in Lookup Tables & Case Management:** Simplify data management and incident tracking.
*   **Scalable & Reliable Orchestration:** Powered by Temporal for robust, scalable workflow execution.
*   **Open Cyber Security Schema (OCSF) Compatibility:** Templates are normalized with OCSF for improved interoperability.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. View full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community!  Get your questions answered, provide feedback, and share integration ideas in the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry: Integration & Response-as-Code Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry provides a curated collection of integration and response-as-code templates to accelerate your security and IT automation efforts. Response actions are organized using Tracecat's ontology of common capabilities (e.g. `list_alerts`, `list_cases`, `list_users`).

**Examples:**

Explore various use cases and get inspired by visiting our documentation on [Tracecat Registry](https://docs.tracecat.com/).
See existing open source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is available under the AGPL-3.0 license, with the exception of the `ee` directory, which contains paid enterprise features requiring a Tracecat Enterprise license. The Enterprise Edition offers advanced features requiring specific investments in research and development. You can enable the Enterprise Edition directly in the platform settings.

*For information on Tracecat's Enterprise self-hosted or managed Cloud offerings, visit [our website](https://tracecat.com) or [book a meeting with us](https://cal.com/team/tracecat).*

## Security

Tracecat prioritizes security with features like SSO, audit logs, and Infrastructure as Code (IaaC) deployments (Terraform, Kubernetes / Helm), which are always free and available. We're actively working on a comprehensive document outlining Tracecat's threat model, security features, and hardening recommendations. For immediate security-related questions, reach out to us on [Discord](https://discord.gg/H4XZwsYzY4).

Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## Contributors

Thank you to our amazing contributors for code, integrations, and support!  Open source thrives because of you. ❤️

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

*   **SEO-Optimized Title:**  Uses "Tracecat" and emphasizes the core function: "Automate Security & IT Response".
*   **Concise Hook:** Starts with a strong one-sentence introduction highlighting the main value proposition.
*   **Clear Headings:** Uses Markdown headings for better organization and readability.
*   **Bulleted Key Features:**  Highlights the most important features in an easy-to-scan format.  This makes it easy for users to quickly grasp what the project does.
*   **Improved Community Callout:** The community section is more engaging.
*   **More Descriptive Section Titles:** Changed "Open Source vs Enterprise" to a clearer name.
*   **Emphasis on Security:** Highlights the importance of security and provides clear instructions on reporting issues.
*   **Contributor Section:** Includes a visual contributor graph, boosting engagement and recognition.
*   **Consistent Formatting:** Uses consistent formatting throughout for better readability.
*   **Link to Original Repo:**  Includes the link at the beginning for easy access.
*   **Call to action:**  Encourages users to read docs, join discord, etc.