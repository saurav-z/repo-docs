# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

**Secure your environment and proactively hunt for threats with the comprehensive resources provided in this repository, a unified hub for Microsoft Sentinel and Microsoft 365 Defender.**

This repository offers a wealth of security content to help you get started with Microsoft Sentinel and provide you security content to secure your environment and hunt for threats. It includes out-of-the-box detections, exploration queries, hunting queries (including Microsoft 365 Defender hunting queries for advanced scenarios), workbooks, playbooks, and more. This is a community-driven project, and your contributions are welcome!

**Key Features:**

*   **Pre-built Detections:** Leverage ready-to-use detection rules to identify and respond to threats quickly.
*   **Hunting Queries:** Proactively search for malicious activities with advanced hunting queries for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Workbooks & Playbooks:** Utilize pre-configured workbooks for visualization and playbooks for automated response actions.
*   **Microsoft 365 Defender Integration:** Benefit from integrated hunting queries designed for advanced threat detection in Microsoft 365 Defender.
*   **Community Driven:** Contribute to and benefit from a community-driven repository.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

**Get Involved & Provide Feedback:**

We value your input and encourage you to participate in the community. Here's how you can connect:

1.  **Q&A for SIEM & SOAR:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **Q&A for XDR:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Submit a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

**Contribution Guidelines:**

This project welcomes contributions. Please review the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md) and refer to this repository's [wiki](https://aka.ms/threathunters) to get started.  

**Detailed Contribution Steps:**
1.  **Fork, Clone, and Branch:** Fork the repository, clone it locally, and create a new branch for your changes.
2.  **Add/Update Contributions:**  Add or update your files (queries, playbooks, etc.) using your preferred method (GitHub Desktop, VS Code, etc.).  Be sure to merge master back to your branch before you push.
3.  **Submit a Pull Request:**  Submit a Pull Request (PR) with details about your proposed changes. Details about the Proposed Changes are required, be sure to include a minimal level of detail so a review can clearly understand the reason for the change and what he change is related to in the code.
4.  **Review & Resolve:** Check the PR for comments, make any necessary changes, and resolve the comments.

**Pull Request Validation:**

The repository includes automated checks to validate the structure of your submissions:

*   **Detection Template Structure Validation:** Ensures all required elements are present in YAML files (e.g., detections).
*   **KQL Validation:** Validates the syntax of KQL queries.  If this check fails go to Azure Pipeline (by pressing on the errors link on the checks tab in your PR)
*   **Detection Schema Validation:** Ensures the schema of a detection is valid.

**Run KQL Validation Locally:**

1.  Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download).
2.  Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` in your terminal.
3.  Run `dotnet test`.

**Run Detection Schema Validation Locally:**

1.  Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download).
2.  Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` in your terminal.
3.  Run `dotnet test`.

**Licensing and Code of Conduct:**

All contributions require a Contributor License Agreement (CLA). This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

**Contribute Today!**

To learn more about what you can contribute and for further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

[Link to Original Repository: Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel)