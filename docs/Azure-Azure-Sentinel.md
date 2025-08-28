# Enhance Your Cybersecurity with Microsoft Sentinel and Microsoft 365 Defender

**This repository is your one-stop shop for security content, offering detections, queries, workbooks, and playbooks to elevate your threat hunting and security posture with Microsoft Sentinel and Microsoft 365 Defender. [Explore the repository](https://github.com/Azure/Azure-Sentinel).**

## Key Features:

*   **Pre-built Detections:** Out-of-the-box detections to quickly identify potential threats.
*   **Hunting Queries:**  Advanced hunting queries tailored for Microsoft 365 Defender and Sentinel, empowering proactive threat hunting.
*   **Exploration Queries:**  Gain insights into your environment with pre-built queries.
*   **Workbooks:** Visualize your security data for better analysis and reporting.
*   **Playbooks:** Automate incident response and streamline security operations.
*   **Unified Security:** Seamlessly integrates Microsoft Sentinel and Microsoft 365 Defender for comprehensive protection.
*   **Community-Driven:** Submit requests for new content or contribute to the community.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved:

We encourage your feedback and contributions! Here’s how to connect and contribute:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit and vote on feature requests on [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** Use the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines:

This project welcomes contributions.  All contributors must agree to a Contributor License Agreement (CLA). See [https://cla.microsoft.com](https://cla.microsoft.com) for details.

**To contribute:**

1.  **Fork & Clone:** Follow [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo)
2.  **Create a Branch:** Make a new branch for your work.
3.  **Add/Update:** Add or update content within the repository.
4.  **Merge & Push:** Be sure to merge master back to your branch before you push.
5.  **Submit a Pull Request (PR):** After pushing your changes, submit a Pull Request (PR). Include details explaining your changes.  Review the [Pull Request](https://github.com/Azure/Azure-Sentinel/pulls) for comments.
6.  **Review:** Make changes as suggested and update your branch. Resolve the comment when done.

### PR Checks

As part of the PR process, the repository runs a structure and KQL validation. If these fail, address the errors. If using custom logs, ensure your schema is defined in the *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables* folder.

**To run KQL validation locally:**

1.  Install the .NET Core 3.1 SDK.
2.  Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
3.  Run `dotnet test`.

**To run Detection Schema Validation locally:**
1. Install the .NET Core 3.1 SDK.
2. Navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
3. Execute `dotnet test`

For more information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.