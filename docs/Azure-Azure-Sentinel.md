# Azure Sentinel and Microsoft 365 Defender: Your Central Hub for Security Content and Threat Hunting

**Secure your environment and proactively hunt for threats with the unified Microsoft Sentinel and Microsoft 365 Defender repository, offering pre-built detections, queries, and more.**

[![GitHub stars](https://img.shields.io/github/stars/Azure/Azure-Sentinel?style=social)](https://github.com/Azure/Azure-Sentinel)

This repository provides a wealth of resources to help you get started with Microsoft Sentinel and Microsoft 365 Defender, including:

*   **Out-of-the-Box Detections:** Quickly identify potential threats.
*   **Exploration Queries:** Gain deeper insights into your data.
*   **Hunting Queries:** Proactively search for malicious activity.
*   **Workbooks:** Visualize data and gain operational insights.
*   **Playbooks:** Automate security tasks and responses.
*   **Microsoft 365 Defender Hunting Queries:** Leverage advanced hunting capabilities.

Explore pre-built content and contribute your own!

[Visit the original repository](https://github.com/Azure/Azure-Sentinel)

## Key Features

*   **Unified Security Content:** Access detections and queries for both Microsoft Sentinel and Microsoft 365 Defender in one place.
*   **Threat Hunting Focus:** Empower security teams with pre-built hunting queries for proactive threat detection.
*   **Community Driven:** Contribute and collaborate with others to improve security content.
*   **Automated Checks:** Ensure code quality and compliance with automated validation processes.
*   **Rich Documentation:** Detailed instructions and resources to guide users and contributors.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved: Feedback & Support

We value your input. Here's how to connect with the community:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and vote on features in the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs (Product or Contribution):** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project welcomes contributions.  Please review these guidelines:

*   **Contributor License Agreement (CLA):**  You must agree to the [Contributor License Agreement (CLA)](https://cla.microsoft.com) before contributing.
*   **Contribution Process:**
    1.  Fork the repo and clone to your local environment.
    2.  Create a branch.
    3.  Make and test your changes (See the tests below)
    4.  Submit a Pull Request (PR) with detailed descriptions.
    5.  Address comments and update the branch as needed.
    6.  Resolve all comments.

### Code Validation and Quality Checks
*   **KQL Validation:**  Ensure KQL queries pass validation.  Run validation locally using `.Net Core 3.1 SDK`:
    1.  Open Shell and navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    2.  Execute `dotnet test`
*   **Detection Schema Validation:** Ensure detection schema complies with guidelines. Run validation locally using `.Net Core 3.1 SDK`:
    1.  Open Shell and navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    2.  Execute `dotnet test`

**See the [wiki](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for in-depth details on contribution and validation.**

### Pull Request Considerations

*   Provide clear details about your changes in the PR.
*   Address comments made during review.
*   Include any new custom logs, and verify table schema is defined in json file in the folder *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables*

### CLA-Bot

The CLA-bot will guide you through the necessary CLA steps.

## Code of Conduct

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For more information and details on contributing, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).