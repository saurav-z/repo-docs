# Enhance Your Security Posture with Microsoft Sentinel & Microsoft 365 Defender

**Protect your environment with pre-built detections, hunting queries, workbooks, and playbooks in the Microsoft Sentinel and Microsoft 365 Defender repository.**  This repository offers a wealth of resources to help you secure your environment and proactively hunt for threats. Explore a collaborative space with resources designed to improve your security posture.

[Link to Original Repo: Microsoft Sentinel and Microsoft 365 Defender](https://github.com/Azure/Azure-Sentinel)

## Key Features

*   **Out-of-the-Box Detections:** Implement immediate security measures with pre-configured detection rules.
*   **Hunting Queries:** Proactively identify threats with advanced hunting queries, including those for Microsoft 365 Defender.
*   **Workbooks:** Visualize data and gain valuable insights with ready-to-use workbooks.
*   **Playbooks:** Automate incident response and streamline security operations.
*   **Unified Security:** Leverage content for both Microsoft Sentinel (SIEM) and Microsoft 365 Defender (XDR).
*   **Community Contributions:** Access and contribute to a community-driven repository of security content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved and Provide Feedback

Your input is valued. Here's how you can contribute and engage:

1.  **Product Q&A for SIEM and SOAR:** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **Product Q&A for XDR:** Join the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

This project welcomes contributions.  To contribute, follow these guidelines:

*   **Contributor License Agreement (CLA):**  Before contributing, you must agree to a Contributor License Agreement (CLA).  Visit https://cla.microsoft.com for details.
*   **Fork and Clone:** Fork the repository and clone it locally.  [General GitHub Fork guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md)
*   **Create a Branch:** Create a new branch for your work.
*   **Make Changes:** Add or update content.
*   **Merge:**  Merge master back into your branch before pushing.
*   **Submit a Pull Request (PR):**  Submit a PR with a detailed description of your changes.  Be sure to include a minimal level of detail so a review can clearly understand the reason for the change and what he change is related to in the code.
*   **PR Review:** Check the PR for comments and address any feedback.

### Pull Request Checks

*   **Detection Template Structure Validation:** PRs undergo validation to ensure the YAML structure of detections is correct.
*   **KQL Validation:** Syntax validation of KQL queries is performed.
*   **Detection Schema Validation:** Automatic validation of detection schema, including frequency, trigger types, and connector IDs.

### Running Validations Locally

*   **KQL Validation:** Install the .NET Core 3.1 SDK and run `dotnet test` in the `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` directory.
*   **Detection Schema Validation:** Install the .NET Core 3.1 SDK and run `dotnet test` in the `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` directory.

For more information on what you can contribute, please refer to the project's [wiki](https://aka.ms/threathunters).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.