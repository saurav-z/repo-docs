# Microsoft Sentinel & Microsoft 365 Defender: Your One-Stop Shop for Security Content

**Enhance your threat detection and response capabilities with the unified repository for Microsoft Sentinel and Microsoft 365 Defender resources.** This repository offers a wealth of pre-built detections, queries, workbooks, and playbooks to secure your environment.  [Explore the original repository](https://github.com/Azure/Azure-Sentinel)

## Key Features

*   **Out-of-the-box Detections:** Implement pre-built detections to identify threats quickly.
*   **Hunting Queries:** Proactively hunt for threats with ready-to-use hunting queries, including those for Microsoft 365 Defender.
*   **Exploration Queries:** Deep dive into your data and understand your security posture.
*   **Interactive Workbooks:** Visualize your security data and gain valuable insights.
*   **Automated Playbooks:** Automate your incident response with pre-built playbooks.
*   **Unified Security Content:**  Resources available for both Microsoft Sentinel and Microsoft 365 Defender.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409) - Official documentation for Microsoft Sentinel.
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide) - Official documentation for Microsoft 365 Defender.
*   [Security Community Webinars](https://aka.ms/securitywebinars) - Stay up-to-date with the latest security trends.
*   [Getting Started with GitHub](https://help.github.com/en#dotcom) -  Learn the basics of using GitHub.

## Get Involved & Contribute

We welcome contributions!  Help improve the repository by:

*   Submitting new detections, queries, and playbooks.
*   Suggesting improvements and feature requests via GitHub Issues.
*   Reporting bugs.

**Contribution Guidelines:**

1.  **Fork the repository:**  Start by forking the repository to your own GitHub account.
2.  **Create a branch:**  Create a new branch for your contributions.
3.  **Make your changes:**  Add or update detections, queries, workbooks, etc.
4.  **Submit a Pull Request (PR):**  Submit a PR with details about your changes for review.

**Important Checks for PRs:**

*   **Detection Template Structure Validation:**  Ensure your YAML files follow the required structure.
*   **KQL Validation:**  Validate the syntax of your KQL queries.
*   **Detection Schema Validation:**  Ensure your detection templates are valid according to schema.

**Run Validations Locally**
*   **KQL Validation:** Run `dotnet test` from `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` after installing .NET Core 3.1 SDK
*   **Detection Schema Validation:** Run `dotnet test` from `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` after installing .NET Core 3.1 SDK

**Code of Conduct**

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For questions or comments, contact [opencode@microsoft.com](mailto:opencode@microsoft.com).

For detailed information and how-to's see the [wiki](https://aka.ms/threathunters).