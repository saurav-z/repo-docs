# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

This repository provides a unified resource for out-of-the-box detections, exploration queries, hunting queries, workbooks, and playbooks to help you proactively defend your environment and hunt for threats using Microsoft Sentinel and Microsoft 365 Defender. [Explore the Azure Sentinel Repository](https://github.com/Azure/Azure-Sentinel).

## Key Features

*   **Pre-built Detections:** Get started quickly with ready-to-use detections to identify potential threats.
*   **Hunting Queries:** Leverage advanced hunting queries for in-depth threat investigations across both Microsoft Sentinel and Microsoft 365 Defender.
*   **Workbooks and Playbooks:** Utilize pre-configured workbooks for data visualization and automated playbooks to streamline your security operations.
*   **Microsoft 365 Defender Integration:** Seamlessly integrated hunting queries for advanced threat detection in Microsoft 365 Defender.
*   **Community-Driven:** Contribute and benefit from a community-driven approach, ensuring a continuously updated and relevant resource.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Community

Your feedback is valuable. Connect with the community and provide feedback through the following channels:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific questions.
2.  **XDR Q&A:** Participate in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related discussions.
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** Report product or contribution bugs by filing a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Provide general feedback on community and contribution processes by filing a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project welcomes contributions. Please review the [Contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries)  to learn how to contribute and the CLA process.

**Contribution Steps:**

1.  **Fork the Repo:** Fork the repository.
2.  **Clone the Repo:** Clone the forked repository to your local machine.
3.  **Create a Branch:** Create a new branch for your changes.
4.  **Make Changes:** Add or update content.
5.  **Merge & Push:** Merge master into your branch, push your changes to GitHub.
6.  **Submit Pull Request:** Submit a Pull Request (PR) for review. Provide details about your changes.
7.  **PR Checks:** Ensure your PR passes all checks. Resolve any issues.

**Pull Request Validation:**  PRs undergo automatic validation checks, including structure and KQL syntax checks.

*   **KQL Validation:**  Ensure the syntax of your KQL queries is valid.
*   **Schema Validation:** Ensure you follow the schema correctly for detections.

**Local KQL Validation:**  You can run KQL and schema validation locally to ensure correct syntax before submitting a Pull Request.

*   **.Net Core 3.1 SDK** is required.
*   Navigate to the appropriate directories and run `dotnet test`.

**[See the project's wiki](https://aka.ms/threathunters) for more detailed information.**

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.