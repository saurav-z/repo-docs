# Microsoft Sentinel and Microsoft 365 Defender Security Content Repository

**Enhance your threat detection and response capabilities with pre-built detections, queries, workbooks, and playbooks for Microsoft Sentinel and Microsoft 365 Defender!**  This repository provides a wealth of security content to help you secure your environment and proactively hunt for threats. ([Back to the original repo](https://github.com/Azure/Azure-Sentinel))

## Key Features:

*   **Pre-built Detections:** Out-of-the-box detections to identify and respond to threats.
*   **Exploration and Hunting Queries:**  Powerful queries for threat hunting and investigating security incidents. Includes queries for Microsoft 365 Defender.
*   **Interactive Workbooks:**  Visualize your security data with pre-configured workbooks.
*   **Automated Playbooks:**  Implement automated responses to security alerts.
*   **Microsoft 365 Defender Integration:**  Includes advanced hunting queries for comprehensive threat detection across Microsoft 365 Defender and Microsoft Sentinel.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved:

We value your feedback and contributions! Here's how you can connect with us and contribute:

*   **Q&A for SIEM and SOAR:** [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
*   **Q&A for XDR:** [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
*   **Feature Requests:** [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
*   **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
*   **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines:

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA).  

**Follow these steps to add your new or updated contributions:**

1.  **Fork and Clone:** Fork the repository and clone it to your local machine.
2.  **Create a Branch:** Create a new branch for your changes.
3.  **Make Changes:** Add or update your contributions using your preferred development tools (GitHub Desktop, Visual Studio, VSCode).
4.  **Merge and Push:**  Merge your changes back to your branch before pushing.
5.  **Submit a Pull Request (PR):** Create a pull request on GitHub. Provide detailed information about your changes.

**Important Checks:**

*   **Pull Request Detection Template Structure Validation:** Ensure your YAML structure adheres to the required format.
*   **Pull Request KQL Validation Check:** Validate your KQL queries for syntax.
*   **Detection Schema Validation:** Validate the schema of your detections.

**Local Validation:**

*   Run KQL validation locally by navigating to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and executing `dotnet test`.
*   Run Detection schema validation locally by navigating to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and executing `dotnet test`.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).