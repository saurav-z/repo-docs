# Microsoft Sentinel and Microsoft 365 Defender Repository

**Enhance your security posture with ready-to-use detections, queries, workbooks, and playbooks for Microsoft Sentinel and Microsoft 365 Defender.** This repository provides a wealth of resources to help you secure your environment and proactively hunt for threats.

[Go to the original repository](https://github.com/Azure/Azure-Sentinel)

**Key Features:**

*   **Out-of-the-box Detections:** Immediately identify and respond to potential threats.
*   **Hunting Queries:** Proactively search for malicious activity within your environment.
*   **Workbooks:** Gain insights into your security posture with interactive visualizations.
*   **Playbooks:** Automate security tasks and incident response processes.
*   **Microsoft 365 Defender Integration:** Leverage advanced hunting queries for comprehensive threat detection across your entire security landscape.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

**Get Involved:**

*   **Feedback:** Share your questions and feedback via:
    *   [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
    *   [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
    *   [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8) for product feature requests
    *   File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=) or [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

**Contribution Guidelines:**

We welcome contributions! Please review the [wiki](https://aka.ms/threathunters) for details on how to contribute.
All contributions must adhere to the [Contributor License Agreement (CLA)](https://cla.microsoft.com).

**Contribution Process:**

1.  **Fork and Clone:** Fork the repository and clone it locally.
2.  **Create a Branch:** Create a new branch for your changes.
3.  **Make Changes:** Add or update content.
4.  **Submit a Pull Request (PR):** Submit your changes for review with detailed descriptions of the changes.

**PR Checks:**

*   **Structure Validation:** Ensures the YAML structure of your contributions is valid.
*   **KQL Validation:** Validates the syntax of your KQL queries.

**Run KQL Validation Locally**

In order to run the KQL validation before submitting Pull Request in you local machine:
* You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
* Execute `dotnet test`

**Run Detection Schema Validation Locally**

In order to run the schema validation before submitting Pull Request in you local machine:
* You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
* Execute `dotnet test`

**Code of Conduct:**

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.