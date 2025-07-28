# Microsoft Sentinel and Microsoft 365 Defender Security Content Repository

**Enhance your security posture and proactively hunt for threats with pre-built detections, queries, and more, all in one place.** This repository, maintained by Microsoft, provides a wealth of resources for Microsoft Sentinel and Microsoft 365 Defender.  Access the original repository: [Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel).

**Key Features:**

*   **Out-of-the-box Detections:** Implement ready-to-use detections to identify potential security threats in your environment.
*   **Exploration & Hunting Queries:** Leverage pre-built KQL queries for threat hunting and security investigations in both Microsoft Sentinel and Microsoft 365 Defender.
*   **Workbooks & Playbooks:** Utilize pre-configured workbooks for data visualization and playbooks to automate incident response.
*   **Microsoft 365 Defender Integration:** Includes hunting queries specifically designed for advanced hunting scenarios in Microsoft 365 Defender.
*   **Community-Driven:**  This repository is open to contributions and welcomes your input to improve its content and effectiveness.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

**Feedback & Support:**

*   **General Product Q&A (SIEM/SOAR):** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
*   **General Product Q&A (XDR):** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
*   **Feature Requests:** Submit or upvote on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
*   **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
*   **Community/Contribution Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

**Contribution Guidelines:**

This project welcomes contributions. Please review the following guidelines before contributing:

*   **Contributor License Agreement (CLA):** You must agree to a Contributor License Agreement (CLA) to contribute.
*   **GitHub Forking:**  Fork the repository, create branches for your work, and submit pull requests.
*   **Contribution Methods:**
    *   Submit directly on the GitHub website by browsing to the target folder and uploading files.
    *   Use GitHub Desktop, Visual Studio, or VS Code to fork, clone, create branches, and push changes.
*   **Pull Request (PR) Requirements:**
    *   Provide detailed descriptions of your proposed changes.
    *   Address any comments received during the review process.
    *   Ensure all required sections for detection YAML templates are included.
*   **Validation Checks:** Pull requests will be validated for KQL syntax and detection schema to ensure the code is working as expected. You can run these validation checks locally using the provided instructions.

*  **KQL validation locally:**
    *  Install .Net Core 3.1 SDK.
    *  Open shell and navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *  Execute `dotnet test`

*  **Schema validation locally:**
    *  Install .Net Core 3.1 SDK.
    *  Open shell and navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *  Execute `dotnet test`

*   **Code of Conduct:** This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For detailed information on contribution guidelines, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section of the [wiki](https://aka.ms/threathunters).