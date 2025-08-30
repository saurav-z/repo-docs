# Azure Sentinel and Microsoft 365 Defender: Enhance Your Security Posture

**Secure your environment and proactively hunt for threats with the comprehensive resources and community support offered by the Azure Sentinel and Microsoft 365 Defender repository.** This repository, maintained by Microsoft, provides a wealth of security content and tools to help you get the most out of these powerful security solutions. ([Original Repo](https://github.com/Azure/Azure-Sentinel))

## Key Features:

*   **Out-of-the-Box Detections:** Leverage pre-built detections to identify and respond to potential threats quickly.
*   **Exploration & Hunting Queries:** Utilize pre-built queries designed for security investigations and threat hunting within your environment.
*   **Microsoft 365 Defender Integration:** Access and deploy advanced hunting queries specifically for Microsoft 365 Defender to bolster your threat detection capabilities.
*   **Workbooks & Playbooks:** Utilize pre-built workbooks and playbooks to improve incident response and automation.
*   **Community-Driven Content:** Benefit from a collaborative environment with the ability to contribute and request new samples and resources.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback:

Your feedback is valuable. Here's how to connect:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for general product questions.
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR discussions.
3.  **Feature Requests:** Submit and vote on feature requests in the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community/Contribution Feedback:** Share feedback with a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines:

We welcome contributions! To contribute:

1.  **Review Contribution Guidelines:** Familiarize yourself with the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) and [Getting Started](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md)
2.  **Fork and Clone:** Fork the repository and clone it to your local machine.
3.  **Create a Branch:** Create a new branch for your changes.
4.  **Make Changes:** Add or update content using GitHub Desktop, Visual Studio, or VS Code.
5.  **Merge and Push:** Merge your changes back to your branch before pushing.
6.  **Submit Pull Request:** Submit a Pull Request (PR) for review. Include detailed information about your proposed changes.
7.  **Address Feedback:** Respond to comments and make necessary changes.

### Pull Request Checks:

*   **Detection Template Structure Validation:** Ensure all YAML structures meet the requirements.
*   **KQL Validation:** Verify the syntax of your KQL queries. Run the KQL validation locally:
    *   Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
    *   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Execute `dotnet test`
    *   If using custom logs, define your table schema in `Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables`
*   **Detection Schema Validation:** Ensure the schema of a detection matches the requirements. Run the Schema validation locally:
    *   Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
    *   Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Execute `dotnet test`

Once your PR is submitted, a CLA-bot will assist with the Contributor License Agreement (CLA) process.

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For more details on contributions, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).