# Azure Sentinel and Microsoft 365 Defender Security Content Repository

**Enhance your security posture with pre-built detections, hunting queries, and playbooks for Microsoft Sentinel and Microsoft 365 Defender.**  This repository provides a wealth of resources to help you get started with Microsoft Sentinel, secure your environment, and proactively hunt for threats. Access the [original repository](https://github.com/Azure/Azure-Sentinel) for the latest updates and contributions.

## Key Features:

*   **Pre-built Detections:** Implement out-of-the-box detections to identify and respond to threats.
*   **Hunting Queries:**  Utilize exploration and hunting queries, including advanced hunting queries for both Microsoft 365 Defender and Microsoft Sentinel, to proactively uncover malicious activity.
*   **Workbooks:** Leverage pre-built workbooks for data visualization and security insights.
*   **Playbooks:** Automate security responses with ready-to-use playbooks.
*   **Comprehensive Coverage:**  Includes security content to secure your environment and hunt for threats.
*   **Microsoft 365 Defender Integration:** Includes hunting queries compatible with Microsoft 365 Defender's advanced hunting capabilities.
*   **Community Driven:** Contribute to and learn from a community of security professionals.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support:

We value your feedback and are here to assist you.  Please use the following channels to share your questions, suggestions, and bug reports:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Product or Contribution Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community and Contribution Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines:

This project welcomes contributions. To contribute, you must agree to a Contributor License Agreement (CLA).

**Contributing to the repository:**

1.  **Fork the repository.**
2.  **Clone the repository.**
3.  **Create a new branch.**
4.  **Make your changes**
5.  **Push your changes to GitHub.**
6.  **Submit a Pull Request.** Provide detailed information about your proposed changes.
7.  **Address comments.** Review the Pull Request for comments and suggestions. Make updates as requested and resolve the comment when completed.

### Pull Request Validation Checks

Ensure your contribution passes validation checks:

*   **Detection Template Structure Validation:** Verify that all required parts of the YAML structure are included. (see [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how))
*   **KQL Validation Check:** Ensure your KQL queries are syntactically correct.
*   **Detection Schema Validation:** The schema of a detection is automatically validated, including frequency, trigger type, and connector IDs.

**To run KQL validation locally:**

*   Install the .Net Core 3.1 SDK.
*   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` in your terminal.
*   Run `dotnet test`.

**To run Detection Schema Validation locally:**

*   Install the .Net Core 3.1 SDK.
*   Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` in your terminal.
*   Run `dotnet test`.

Upon submitting a pull request, a CLA-bot will assess if you need to sign a CLA. Follow the instructions provided by the bot.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For more information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).