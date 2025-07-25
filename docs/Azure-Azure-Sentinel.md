# Microsoft Sentinel and Microsoft 365 Defender Security Content Repository

**Enhance your security posture and proactively hunt for threats with the comprehensive security content provided in this repository.** This repository offers a centralized hub for detections, queries, workbooks, playbooks, and more to help you maximize the value of Microsoft Sentinel and Microsoft 365 Defender.  [Visit the original repository](https://github.com/Azure/Azure-Sentinel) to get started.

## Key Features

*   **Out-of-the-Box Detections:** Pre-built detections to identify and respond to threats.
*   **Exploration & Hunting Queries:**  Customizable queries for proactive threat hunting.
*   **Workbooks:**  Interactive dashboards for data visualization and analysis.
*   **Playbooks:**  Automated response workflows for incident handling.
*   **Microsoft 365 Defender Integration:** Advanced hunting queries for comprehensive security across Microsoft 365.
*   **Community Contributions:**  Share and access resources from the security community.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback & Support

We value your feedback and welcome your questions.  Engage with the community through the following channels:

1.  **Microsoft Sentinel Tech Community:** [Join the conversation](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **Microsoft 365 Defender Tech Community:** [Join the conversation](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:**  Submit and vote on feature requests via the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:**  File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:**  File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

This project encourages contributions to enhance its capabilities.

### Contribution Process

1.  **Fork and Clone:**  Fork the repository and clone it locally.
2.  **Create a Branch:**  Create a new branch for your changes.
3.  **Make Changes:**  Add or update content using your preferred tools (GitHub Desktop, VS Code, etc.).
4.  **Merge Master (Optional):** Merge master into your branch to incorporate the latest updates.
5.  **Push Changes:**  Push your branch to your forked repository.
6.  **Submit a Pull Request (PR):**  Submit a PR detailing your proposed changes.

### Important Checks and Validations

*   **Detection Template Structure Validation:** The PR process includes checks to ensure the YAML structure of detection templates is valid. Review the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for details and required sections.
*   **KQL Validation:**  The PR process validates the syntax of KQL queries within your templates. If validation fails, correct the errors and review the output in the Azure Pipeline (accessed via the PR Checks tab). Custom table schema must be defined as shown in the original documentation.
*   **Detection Schema Validation:** The schema of the detection (period, trigger type, connectors Ids, etc.) is validated.  Follow error messages to correct any schema errors.

### Running Validations Locally

*   **KQL Validation:**
    1.  Install the .NET Core 3.1 SDK.
    2.  Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` in your terminal.
    3.  Run `dotnet test`.
*   **Detection Schema Validation:**
    1.  Install the .NET Core 3.1 SDK.
    2.  Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` in your terminal.
    3.  Run `dotnet test`.

### Contributor License Agreement (CLA)

All contributions require you to agree to a Contributor License Agreement (CLA).  Follow the instructions from the CLA-bot when submitting your pull request.

### Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For further details on contributing and available content, please consult the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).