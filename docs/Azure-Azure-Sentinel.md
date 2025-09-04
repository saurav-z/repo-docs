# Azure Sentinel & Microsoft 365 Defender Security Content Repository

**Enhance your security posture with the unified repository offering detections, hunting queries, workbooks, and more for Microsoft Sentinel and Microsoft 365 Defender!** This repository serves as a comprehensive resource for security professionals, providing out-of-the-box content to secure your environment and proactively hunt for threats.

[Link to Original Repository](https://github.com/Azure/Azure-Sentinel)

## Key Features

*   **Pre-built Detections:** Leverage readily available detections to identify and respond to threats quickly.
*   **Hunting Queries:** Explore detailed hunting queries for both Microsoft Sentinel and Microsoft 365 Defender for advanced threat hunting.
*   **Workbooks & Playbooks:** Utilize pre-built workbooks for data visualization and actionable playbooks to automate your security workflows.
*   **Unified Security:** Designed for use with both Microsoft Sentinel and Microsoft 365 Defender for a holistic security approach.
*   **Community Driven:** Contribute and benefit from community-contributed content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)

## Feedback and Support

We value your feedback! Connect with the community and provide input through these channels:

1.  **SIEM & SOAR Q&A:** [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **Community & Contribution Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

This project welcomes contributions. Please review the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) and the [wiki](https://aka.ms/threathunters) for details.

### Contribution Process

1.  **Fork & Clone:** Fork the repository and clone it to your local environment.
2.  **Create a Branch:** Create a new branch for your contributions.
3.  **Make Changes:** Add or update content in your branch.
4.  **Merge Master:** Before pushing your branch, merge the master branch back into your branch.
5.  **Submit a Pull Request:** Submit a pull request with detailed information about your changes.

### Pull Request Checks

*   **Detection Template Structure Validation:** Ensure your YAML files adhere to the required structure.
*   **KQL Validation:** Verify the syntax of your KQL queries.
*   **Detection Schema Validation** Verify format and required attributes.

#### Running Validation Locally

*   **KQL Validation:**  Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and run `dotnet test`.  Custom table schema must be defined in  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables`
*   **Detection Schema Validation:** Navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and run `dotnet test`.

### Contribution License Agreement (CLA)

All contributions require you to agree to a Contributor License Agreement (CLA).  Follow the instructions provided by the CLA-bot when submitting your pull request.

## Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For further details, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).