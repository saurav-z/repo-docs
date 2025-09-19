# Microsoft Sentinel & Microsoft 365 Defender Security Content Repository

**Enhance your security posture with pre-built detections, hunting queries, and more for Microsoft Sentinel and Microsoft 365 Defender, all in one place.** This repository, maintained by Microsoft, provides a wealth of security content to help you detect threats and secure your environment. ([Back to the original repo](https://github.com/Azure/Azure-Sentinel))

**Key Features:**

*   **Out-of-the-Box Detections:** Pre-configured detections to identify potential threats.
*   **Hunting Queries:**  Explore your data with ready-to-use queries for proactive threat hunting.
*   **Microsoft 365 Defender Integration:** Includes advanced hunting queries for comprehensive security analysis across your Microsoft 365 environment.
*   **Workbooks & Playbooks:**  Utilize pre-built workbooks and playbooks for enhanced automation and data visualization.
*   **Community Driven:**  Contribute and share your own security content to benefit the community.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support

We value your feedback. Here's how you can get help:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit or upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We welcome contributions! Please review these guidelines before submitting your work.

**Contribution Process:**

1.  **Contributor License Agreement (CLA):** You must agree to a CLA to contribute.  See [https://cla.microsoft.com](https://cla.microsoft.com) for details.
2.  **Fork & Clone:** Fork the repository and clone it locally.
3.  **Create a Branch:** Create a new branch for your changes.
4.  **Make Changes:** Add or update content in your branch.
5.  **Merge Master:**  Merge the latest changes from `master` into your branch before pushing.
6.  **Push and Create Pull Request:** Push your changes and submit a Pull Request (PR).

**Pull Request Requirements:**

*   **Detailed Description:**  Provide a clear explanation of your changes, including the reason for the change.
*   **Validation Checks:**  PRs undergo automated checks, including:
    *   **Detection Template Structure Validation:** Ensures the correct YAML structure for detections.
    *   **KQL Validation:** Validates the syntax of your KQL queries. See below on how to run these locally.
*   **Review and Feedback:**  Be prepared to address comments and make revisions as requested.
*   **KQL Validation locally:**
    *   Install .Net Core 3.1 SDK ([How to download .Net](https://dotnet.microsoft.com/download))
    *   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Execute `dotnet test`
*   **Detection Schema Validation locally:**
    *   Install .Net Core 3.1 SDK ([How to download .Net](https://dotnet.microsoft.com/download))
    *   Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Execute `dotnet test`

**Contribution Resources:**

*   For more information on what you can contribute and further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For any questions or comments, contact [opencode@microsoft.com](mailto:opencode@microsoft.com).