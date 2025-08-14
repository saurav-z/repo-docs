# Microsoft Sentinel and Microsoft 365 Defender Community Resources

**Enhance your security posture and proactively hunt threats with the official Microsoft Sentinel and Microsoft 365 Defender community repository.**  This repository provides valuable resources for security professionals, including out-of-the-box detections, hunting queries, workbooks, and more.  

[Visit the original repository](https://github.com/Azure/Azure-Sentinel)

## Key Features:

*   **Pre-built Detections:** Deploy ready-to-use detections to identify and respond to threats.
*   **Hunting Queries:** Proactively search for threats in your environment with both Microsoft Sentinel and Microsoft 365 Defender hunting queries.
*   **Workbooks and Playbooks:** Utilize pre-built workbooks for data visualization and playbooks for automated response.
*   **Community-Driven Content:** Benefit from contributions and suggestions from the security community.
*   **Unified Security:** Get unified security content across Microsoft Sentinel and Microsoft 365 Defender.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)

## Community and Feedback

We value your feedback and contributions.

*   **General Q&A (SIEM & SOAR):** [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
*   **General Q&A (XDR):** [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
*   **Feature Requests:** [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
*   **Report Bugs:** File a GitHub Issue using [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
*   **Community Feedback:** File a GitHub Issue using [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contributing

This project welcomes contributions!  Please review the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md) and the [wiki](https://aka.ms/threathunters) to get started.

### Contribution Steps
1.  Fork the repository and clone it locally.
2.  Create a new branch for your changes.
3.  Add/update your content.
4.  Push your changes to your fork.
5.  Submit a pull request.
6.  Address any comments from the reviewers.

### PR Validation
*   **Detection Template Structure Validation Check**:  Ensure all required sections are included.
*   **KQL Validation Check**:  Verify your KQL queries are valid.
*   **Detection Schema Validation Tests:** Ensure correct format and attributes.

*   Run KQL validation before submitting Pull Request in you local machine:
    *   Ensure **.Net Core 3.1 SDK** installed
    *   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Execute `dotnet test`

*   Run Detection Schema Validation before submitting Pull Request in you local machine:
    *   Ensure **.Net Core 3.1 SDK** installed
    *   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Execute `dotnet test`

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.