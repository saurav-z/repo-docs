# Microsoft Sentinel & Microsoft 365 Defender: Your Central Hub for Security Content & Threat Hunting

**Enhance your security posture with the comprehensive Microsoft Sentinel and Microsoft 365 Defender repository, providing a wealth of detections, queries, workbooks, and playbooks.** This repository offers invaluable resources to secure your environment and proactively hunt for threats.  ([Original Repo](https://github.com/Azure/Azure-Sentinel))

## Key Features:

*   **Out-of-the-Box Detections:**  Implement pre-built detection rules to identify and respond to threats quickly.
*   **Exploration and Hunting Queries:**  Leverage ready-to-use queries for in-depth investigation and threat hunting.
*   **Microsoft 365 Defender Integration:** Includes advanced hunting queries specifically designed for Microsoft 365 Defender.
*   **Workbooks and Playbooks:**  Utilize pre-configured workbooks for data visualization and playbooks to automate security tasks.
*   **Community Driven:**  Access a collaborative platform, submit feature requests, and contribute your own security content.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback & Support:

We value your input and are here to assist you:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific questions.
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related discussions.
3.  **Feature Requests:**  Submit or upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Bug Reports:**  Report product or contribution bugs via a [GitHub Issue using the Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community Feedback:**  Provide general feedback using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contributing Guidelines:

This project welcomes contributions!  Please review the following guidelines:

### Contribution License Agreement (CLA):

*   All contributions require a Contributor License Agreement (CLA). See [https://cla.microsoft.com](https://cla.microsoft.com) for details.

### Contribution Process:

1.  **Fork the repository.**
2.  **Create a branch** for your changes.
3.  **Make your additions/updates.**
4.  **Submit a Pull Request (PR)** with detailed information about your changes.  Be sure to include a minimal level of detail so a review can clearly understand the reason for the change and what he change is related to in the code.
5.  **Address any feedback** from the review process.

### Pull Request Checks:

*   **Detection Template Structure Validation:** PRs automatically check the YAML structure for required sections, especially for detections.
*   **KQL Validation:**  The KQL queries in your templates will be syntax-checked. If this fails, use the Azure Pipeline error link in the Checks tab to review the specific error.
*   **Detection Schema Validation:** An automatic validation of the schema of a detection will occur.

### Running Validations Locally:

*   **KQL Validation:**
    *   Install .NET Core 3.1 SDK
    *   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Run `dotnet test`
*   **Detection Schema Validation:**
    *   Install .NET Core 3.1 SDK
    *   Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Run `dotnet test`

### Additional Information:

*   For specific contribution details, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).
*   This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.