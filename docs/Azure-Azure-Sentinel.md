# Microsoft Sentinel and Microsoft 365 Defender Security Content Repository

Get started with Microsoft Sentinel and Microsoft 365 Defender by accessing out-of-the-box detections, hunting queries, workbooks, and playbooks to secure your environment and proactively hunt for threats. 

[Link to original repo: https://github.com/Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel)

**Key Features:**

*   **Out-of-the-Box Detections:** Pre-built detections to identify and respond to threats.
*   **Hunting Queries:**  Proactive threat hunting with Microsoft Sentinel and Microsoft 365 Defender. Includes advanced hunting queries for in-depth threat analysis.
*   **Workbooks:**  Interactive dashboards and visualizations for security insights.
*   **Playbooks:**  Automated response actions to streamline incident handling.
*   **Extensive Security Content:** A wide range of resources to get you ramped up quickly.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

**Feedback and Support:**

Your feedback is valuable. Here are ways to ask questions or provide suggestions:

1.  **SIEM and SOAR Q&A:**  Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:**  Submit feature requests via the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

**Contribution Guidelines:**

This project welcomes contributions.  All contributors must agree to the [Contributor License Agreement (CLA)](https://cla.microsoft.com).

**How to Contribute:**

1.  **Get Started:**  Refer to the [wiki](https://aka.ms/threathunters) for detailed instructions.
2.  **Fork the Repo:** Fork the repository on GitHub.
3.  **Clone the Repo:**  Clone the forked repository to your local machine using your favorite method such as [GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop) or [Visual Studio](https://visualstudio.microsoft.com/vs/) or [VSCode](https://code.visualstudio.com/?wt.mc_id=DX_841432)
4.  **Create a Branch:** Create a new branch for your changes.
5.  **Make Changes:**  Add or update security content.
6.  **Merge Master:** Merge master back to your branch before submitting.
7.  **Submit a Pull Request:**  Submit your changes for review.

**Pull Request Details:**

*   Clearly describe the changes and the reason for them.
*   Be sure to include a minimal level of detail so a review can clearly understand the reason for the change and what he change is related to in the code.

**PR Checks and Validation:**

*   **Detection Template Structure Validation:**  Ensures your detection templates follow the required YAML structure.  See [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for details.
*   **KQL Validation Check:**  Validates the syntax of your KQL queries. If a validation fails, review the [Azure Pipeline](.github/Media/Azurepipeline.png) for details.
*   **Detection Schema Validation:**  Automated validation of detection schemas, ensuring frequency, trigger types, connectors, etc. are correctly formatted.

**Running KQL and Schema Validation Locally:**

To run validations locally:

*   **KQL Validation:**  Navigate to the KqlvalidationsTests directory and execute `dotnet test`.
*   **Detection Schema Validation:** Navigate to the DetectionTemplateSchemaValidation directory and execute `dotnet test`.

**Additional Resources:**

*   [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
*   [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
*   Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) for questions about the Code of Conduct.