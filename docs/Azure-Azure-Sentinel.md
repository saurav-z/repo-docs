# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

**Stay ahead of threats with the unified Microsoft Sentinel and Microsoft 365 Defender repository, offering a wealth of security content for proactive threat hunting and incident response.** This repository provides a comprehensive collection of detections, exploration queries, hunting queries, workbooks, and playbooks to help you optimize your security operations.

[Link to Original Repo: https://github.com/Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel)

## Key Features

*   **Out-of-the-Box Detections:** Implement pre-built rules to identify and respond to threats quickly.
*   **Hunting Queries:** Proactively search for malicious activities within your environment.
*   **Exploration Queries:** Gain deeper insights into your data for analysis and investigation.
*   **Workbooks:** Visualize your security data and gain actionable intelligence.
*   **Playbooks:** Automate your incident response workflows.
*   **Microsoft 365 Defender Integration:** Leverages advanced hunting capabilities for comprehensive threat detection across your environment.
*   **Community-Driven:** Contribute to and benefit from community-generated content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved

We encourage your feedback and contributions! Here's how you can engage:

1.  **Product Q&A for SIEM/SOAR:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **Product Q&A for XDR:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and upvote ideas in the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project welcomes contributions. All contributors must agree to the Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

### How to Contribute

1.  **Fork and Clone:** Fork the repository and clone it locally.
2.  **Create a Branch:** Create a branch for your changes.
3.  **Make Changes:** Add or update your contributions.
4.  **Merge and Push:** Merge master back to your branch and push your changes to GitHub.
5.  **Submit a Pull Request:** Create a pull request for review.  Details about the proposed changes are required.

### Pull Request Validation

*   **Detection Template Structure Validation:** The system validates the structure of detection templates. Ensure all required sections are included.
*   **KQL Validation:**  The system validates the syntax of KQL queries.
    *   If you are using custom logs tables, verify your table schema is defined in a JSON file in the appropriate directory.
    *   To run the KQL validation locally, install .NET Core 3.1 SDK, navigate to the KQL validation tests directory and run `dotnet test`.
*   **Detection Schema Validation:** Automatic validation of the detection schema, including frequency, trigger type, and connector Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)).
    *   To run the Detection Schema validation locally, install .NET Core 3.1 SDK, navigate to the Detection Template Schema Validation tests directory and run `dotnet test`.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

For more information on what you can contribute and further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.