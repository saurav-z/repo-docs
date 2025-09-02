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

## **Tracecat: Automate Security and IT Workflows with Open Source**

Tracecat is an open-source automation platform designed to streamline security and IT response engineering, offering YAML-based templates and a no-code UI for efficient workflow creation. **([Check out the original repo!](https://github.com/TracecatHQ/tracecat))**

### **Key Features**

*   **YAML-Based Templates:** Simplify integrations with easy-to-use, YAML-based templates.
*   **No-Code UI for Workflows:** Build and manage workflows effortlessly with an intuitive user interface.
*   **Built-in Lookup Tables & Case Management:** Leverage built-in tools for efficient data management and incident tracking.
*   **Scalable Orchestration:** Powered by Temporal for reliable and scalable workflow execution.
*   **Open Cyber Security Schema (OCSF) Compliance:** Template inputs normalized to fit the OCSF ontology.

![Tracecat workflow](/img/workflow.png)

## **Getting Started**

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### **Run Tracecat Locally**

Deploy a local Tracecat stack using Docker Compose. View full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### **Run Tracecat on AWS Fargate**

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. View full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### **Run Tracecat on Kubernetes**

Coming soon.

## **Community**

Join the Tracecat community to ask questions, share feedback, and propose new integration ideas. Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## **Tracecat Registry: Integration & Response Templates**

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry is a collection of integration and response-as-code templates, organized by a common ontology of capabilities.

*   **Explore Templates:** Visit our documentation for use cases and ideas.
*   **Open Source Templates:** Check out existing open-source templates in our [repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## **Open Source vs. Enterprise**

This repository is available under the AGPL-3.0 license, excluding the `ee` directory, which contains features requiring a Tracecat Enterprise license. Enterprise features require specific R&D investment.

*For information about Tracecat's Enterprise self-hosted or managed Cloud offering, visit [our website](https://tracecat.com) or [book a meeting with us](https://cal.com/team/tracecat).*

## **Security**

SSO, audit logs, and IaaC deployments (Terraform, Kubernetes / Helm) are always free. We're creating a comprehensive list of threat models, security features, and hardening recommendations. For immediate answers, reach out to us on [Discord](https://discord.gg/H4XZwsYzY4).

Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## **Contributors**

Thank you to all our contributors for your code, integrations, and support.

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>
```
Key improvements and optimizations:

*   **SEO-Optimized Title and Hook:** Added a clear, concise title and a one-sentence hook to grab attention and improve search ranking.
*   **Clear Headings:** Structured the README with clear headings for each section.
*   **Bulleted Key Features:** Used bullet points to highlight the core functionalities, making them easy to scan.
*   **Call to Action:** Added links to documentation, community, and templates to encourage engagement.
*   **Keywords:** Incorporated relevant keywords like "security automation," "IT automation," "open source," and "workflow."
*   **Concise Language:** Streamlined the descriptions to be more direct and easier to understand.
*   **Focus on Benefits:** Highlighted the advantages of using Tracecat (e.g., efficiency, ease of use, scalability).
*   **Community Emphasis:** Clearly highlighted the community resources (Discord).
*   **Contributor Section:** Retained the contributor section with the visual representation.
*   **Original Repo Link:** Added a link back to the original repo at the beginning to satisfy the prompt.