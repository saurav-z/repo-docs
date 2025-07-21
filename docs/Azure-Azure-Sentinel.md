# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender 

This repository is your central hub for security content, offering a wealth of detections, queries, workbooks, and playbooks to fortify your environment with Microsoft Sentinel and Microsoft 365 Defender. Explore the original repo at [https://github.com/Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel)

**Key Features:**

*   **Out-of-the-Box Detections:** Implement pre-built detections to proactively identify threats.
*   **Exploration and Hunting Queries:** Leverage insightful queries to investigate security incidents and uncover hidden threats.
*   **Microsoft 365 Defender Integration:** Access advanced hunting queries for comprehensive threat detection across Microsoft 365 Defender and Microsoft Sentinel.
*   **Workbooks & Playbooks:** Utilize ready-to-use workbooks for data visualization and automation to streamline incident response.
*   **Community-Driven:** Contribute your own detections, queries, and more to help improve your organization's security posture.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

**Feedback and Support:**

We value your input! Here's how to connect with the community:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit your ideas on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Submit a feature request using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

**Contribution Guidelines:**

We welcome contributions! Please review the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md) and the [wiki](https://aka.ms/threathunters) to get started. Contributions require a Contributor License Agreement (CLA).

**Steps to Contribute:**

1.  **Fork the Repository:** Fork the repository to your GitHub account.
2.  **Clone the Repository:** Clone the forked repository to your local machine.
3.  **Create a Branch:** Create a new branch for your changes.
4.  **Make Changes:** Add or update detections, queries, workbooks, etc.
5.  **Commit and Push:** Commit your changes and push them to your branch on GitHub.
6.  **Create a Pull Request:** Submit a Pull Request to the main repository.
7.  **Review and Address Feedback:** Respond to any comments and make necessary updates.

**Pull Request Checks:**

*   **Structure Validation:** Ensures the correct YAML structure for detections.
*   **KQL Validation:** Validates the syntax of KQL queries.
*   **Schema Validation:** Validates the detection's trigger type and threshold.

**Local Validation:**
*   **KQL Validation:** Run `dotnet test` from the  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` directory.
*   **Schema Validation:** Run `dotnet test` from the `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` directory.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.