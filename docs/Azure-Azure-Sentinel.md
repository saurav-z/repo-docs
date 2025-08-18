# Azure Sentinel & Microsoft 365 Defender: Your Central Hub for Cybersecurity

**Secure your environment and proactively hunt for threats with the unified resources of Microsoft Sentinel and Microsoft 365 Defender, all available in this collaborative repository.**  [Click here to view the original repository.](https://github.com/Azure/Azure-Sentinel)

This repository provides a wealth of security content to help you get the most out of Microsoft Sentinel and Microsoft 365 Defender. Benefit from:

*   **Ready-to-Use Detections:** Leverage pre-built detections to identify potential threats in your environment.
*   **Exploration Queries:** Quickly analyze your data with pre-configured queries.
*   **Hunting Queries:** Proactively search for threats with advanced hunting capabilities.
*   **Workbooks:** Visualize your security data with pre-built and customizable workbooks.
*   **Playbooks:** Automate your security responses with pre-built playbooks.
*   **Microsoft 365 Defender Integration:** Access hunting queries specifically designed for advanced hunting scenarios within Microsoft 365 Defender.

## Key Features

*   **Unified Security Content:** Access a wide range of security content for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Community Driven:** Contribute to and benefit from a community-driven repository.
*   **Easy Onboarding:** Get ramped up quickly with pre-built content and resources.
*   **Continuous Updates:** Stay ahead of threats with regularly updated detections, queries, and more.
*   **Advanced Hunting:** Leverage hunting queries for in-depth threat investigation.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback

We value your input! Here's how to connect with the community:

1.  **General Product Q&A (SIEM & SOAR):** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **General Product Q&A (XDR):** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Product Feature Requests:** Submit or upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Product or Contribution Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback on Community & Contribution:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

This project welcomes contributions! To contribute, please review the following guidelines:

1.  **Contributor License Agreement (CLA):**  Before contributing, you must agree to the [CLA](https://cla.microsoft.com/).
2.  **Contribution Process:**

    *   **Fork, Clone, and Branch:** Follow standard GitHub practices to fork, clone, and create a branch for your changes.
    *   **Make Changes:**  Add your new or updated content.
    *   **Merge Master:** Be sure to merge master back to your branch before you push.
    *   **Submit Pull Request:** Create a Pull Request (PR) on GitHub.
    *   **Provide Details:** Include a clear explanation of the changes and why they are being made.
    *   **Review PR:** Review comments and make any necessary changes.

### Pull Request Checks and Validation

*   **YAML Structure Validation:** Ensure your YAML files for detections adhere to the required structure.
*   **KQL Validation:** All KQL queries will be validated for syntax errors.
*   **Detection Schema Validation:** Ensure your detection schemas are valid based on the detection type, frequency, and trigger settings.

### Run Validation Locally

You can run the KQL and Schema validation locally before submitting a Pull Request:

*   **Install .NET Core 3.1 SDK:**  [Download .NET](https://dotnet.microsoft.com/download)
*   **KQL Validation:** Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and execute `dotnet test`.
*   **Schema Validation:** Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and execute `dotnet test`.

**For detailed contribution steps and further information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).**

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.