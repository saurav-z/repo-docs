# Enhance Your Security Posture with Microsoft Sentinel & Microsoft 365 Defender

This repository provides a wealth of resources to help you proactively secure your environment, including out-of-the-box detections, hunting queries, workbooks, and playbooks for Microsoft Sentinel and Microsoft 365 Defender.  [Explore the Azure Sentinel Repository](https://github.com/Azure/Azure-Sentinel).

## Key Features

*   **Pre-built Detections:** Implement ready-to-use security detections to identify threats quickly.
*   **Hunting Queries:** Proactively search for threats within your environment.
*   **Comprehensive Security Content:**  Benefit from workbooks and playbooks designed to streamline security operations.
*   **Microsoft 365 Defender Integration:** Leverage advanced hunting queries across Microsoft 365 Defender and Sentinel.
*   **Community-Driven:** Contribute to and benefit from a community-driven repository.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved: Contribute and Provide Feedback

We value your contributions and feedback to help us make this repository better.

*   **General Product Questions (SIEM & SOAR):** [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
*   **General Product Questions (XDR):** [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
*   **Product Feature Requests:** [Microsoft Sentinel Feedback Forum](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
*   **Report Bugs:** File a [Bug Report](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
*   **General Feedback:** File a [Feature Request](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We welcome contributions!  Please review the following guidelines:

1.  **Contributor License Agreement (CLA):** You must agree to a CLA to contribute.  See [https://cla.microsoft.com](https://cla.microsoft.com) for details.
2.  **Contribution Process:**

    *   Fork and clone the repository or directly upload files to a specific folder.
    *   Create a branch for your changes.
    *   Make your additions/updates.
    *   Test queries with KQL and Detection Template Schema Validation locally before submitting a Pull Request
    *   Submit a Pull Request with clear details about the changes.
    *   Address any comments from reviewers.

    **For detailed contribution steps, see the [wiki](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how).**

###  KQL Validation
*   The KQL validation validates the KQL queries defined in the template.
*   If this check fails go to Azure Pipeline (by pressing on the errors link on the checks tab in your PR)
*   If you are using custom logs table (a table which is not defined on all workspaces by default) you should verify your table schema is defined in json file in the folder *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables*
*   In order to run the KQL validation before submitting Pull Request in you local machine:
    *   You need to have .Net Core 3.1 SDK installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
    *   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Execute `dotnet test`

###  Detection Schema Validation
*   The schema validation includes the detection's frequency and period, the detection's trigger type and threshold, validity of connectors Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)), etc.
*   In order to run the KQL validation before submitting Pull Request in you local machine:
    *   You need to have .Net Core 3.1 SDK installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
    *   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Execute `dotnet test`

3.  **Pull Request Requirements:**  Your PR must include a detailed description of the proposed changes.
4.  **PR Checks:** PRs are automatically checked for structure and KQL validation.
5.  **Code of Conduct:** This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
6.  **Get Started:** For more detailed contribution information, including what you can contribute, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section of the project's [wiki](https://aka.ms/threathunters).

Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions or comments regarding the Code of Conduct.