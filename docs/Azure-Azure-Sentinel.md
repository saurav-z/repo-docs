# Azure Sentinel & Microsoft 365 Defender: Security Resources and Community Contributions

Safeguard your environment and proactively hunt for threats with the comprehensive resources available in the Azure Sentinel and Microsoft 365 Defender repository.  [Explore the Repository](https://github.com/Azure/Azure-Sentinel).

## Key Features

*   **Out-of-the-Box Detections:** Implement pre-built detections to identify potential security threats.
*   **Exploration & Hunting Queries:** Utilize ready-made queries for in-depth analysis and threat hunting.
*   **Workbooks & Playbooks:** Leverage pre-configured workbooks for data visualization and playbooks for automated incident response.
*   **Microsoft 365 Defender Integration:**  Includes hunting queries specifically for advanced hunting in both Microsoft 365 Defender and Microsoft Sentinel.
*   **Community Driven:**  Contribute and access community-contributed security content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Community

We value your input! Here's how to get involved:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Participate in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Submit a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project welcomes contributions! Before contributing, review the following:

*   **Contribution License Agreement (CLA):**  All contributions require agreement to a CLA. Visit [https://cla.microsoft.com](https://cla.microsoft.com) for details.
*   **Getting Started:** For first-time contributors, review the [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) before cloning or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
*   **Contribution Methods:**
    *   Submit directly on the GitHub website by uploading files and creating a pull request.
    *   Use GitHub Desktop, Visual Studio, or VSCode to fork, clone, branch, make changes, merge, and push your contributions.
*   **Pull Request (PR) Process:**
    *   Submit a pull request after pushing changes.
    *   Provide detailed descriptions of changes.
    *   Address comments and resolve issues.
    *   Ensure PRs pass the PR checks, including structure and KQL validation.

### Pull Request Validation Checks

*   **Detection Template Structure Validation:** Ensures all required parts of the YAML structure are included.
*   **KQL Validation:** Validates the syntax of KQL queries.

    *   If custom logs table is used, the schema must be defined in JSON file in the folder *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables*
*   **Detection Schema Validation:** Validates the schema of a detection, including its frequency and period, trigger type and threshold, and the validity of connectors.
### Run Validation Locally

*   **KQL Validation:** Run `dotnet test` from `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` after installing the [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
*   **Detection Schema Validation:** Run `dotnet test` from `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` after installing the [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)

### Code of Conduct

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For more details on contributing, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).