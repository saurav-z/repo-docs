# **Microsoft Sentinel & Microsoft 365 Defender: Your Unified Security Content Hub**

This repository provides a wealth of detections, queries, workbooks, and more to help you proactively secure your environment with Microsoft Sentinel and Microsoft 365 Defender.  [Explore the Azure Sentinel GitHub Repository](https://github.com/Azure/Azure-Sentinel).

## Key Features

*   **Out-of-the-Box Detections:** Pre-built security detections to identify and respond to threats quickly.
*   **Hunting Queries:**  Proactive hunting queries for both Microsoft Sentinel and Microsoft 365 Defender, including advanced hunting scenarios.
*   **Exploration Queries:**  Pre-built queries to explore your data.
*   **Workbooks:** Interactive workbooks for data visualization and security analysis.
*   **Playbooks:** Automated response playbooks to streamline incident handling.
*   **Microsoft 365 Defender Integration:** Includes hunting queries specifically for Microsoft 365 Defender's advanced threat hunting capabilities.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Feedback & Support

We value your input. Connect with us through these channels:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit or upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Bug Reports:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **Contribution & Community Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We welcome contributions! To get started:

*   Refer to this repository's [wiki](https://aka.ms/threathunters).
*   Review the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
*   Contributions require a Contributor License Agreement (CLA).
*   For detailed steps on how to contribute, refer to the [wiki](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how).

### Pull Request (PR) Checks

*   **Structure Validation:**  PRs are checked to ensure the correct YAML structure.
*   **KQL Validation:**  KQL queries are checked for syntax. If the query uses a custom log table, make sure its schema is defined in a JSON file within the custom tables directory
*   **Detection Schema Validation**: Schema validation includes format and attributes like frequency, period, trigger type, connectors IDs, etc.

### Running Validations Locally
You will need to install the .NET Core 3.1 SDK to run the KQL and Detection schema validations locally.

*   **KQL Validation:** Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and run `dotnet test`.
*   **Detection Schema Validation:** Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and run `dotnet test`.