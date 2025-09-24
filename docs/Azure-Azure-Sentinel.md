# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

This repository is your one-stop-shop for pre-built security content to strengthen your environment with Microsoft Sentinel and Microsoft 365 Defender.  [Check it out on GitHub!](https://github.com/Azure/Azure-Sentinel)

**Key Features:**

*   **Ready-to-Use Detections:** Get immediate value with pre-configured detections to identify threats.
*   **Hunting Queries:** Proactively seek out malicious activity with advanced hunting queries for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Exploration Queries:** Quickly understand your data and uncover hidden insights with helpful exploration queries.
*   **Workbooks and Playbooks:** Automate tasks, visualize data, and streamline your security operations with pre-built workbooks and playbooks.
*   **Microsoft 365 Defender Integration:** Leverage advanced hunting scenarios and insights from Microsoft 365 Defender.
*   **Community Driven:** This repository welcomes contributions from the community to enhance security content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback & Support

We value your feedback! Here's how you can connect:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community/Contribution Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribute

We welcome contributions! Please review the following guidelines:

1.  **Contributor License Agreement (CLA):**  All contributions require agreement to a CLA. Visit [https://cla.microsoft.com](https://cla.microsoft.com) for details.
2.  **Contribution Methods:**
    *   **Directly on GitHub:** Upload files directly to the desired folder and submit a Pull Request.
    *   **Using GitHub Desktop/Visual Studio/VSCode:**
        *   Fork the repository.
        *   Clone the repository.
        *   Create a new branch.
        *   Make your changes.
        *   Merge master back into your branch before pushing.
        *   Push your changes.
        *   Submit a Pull Request (PR).
3.  **Pull Request (PR) Requirements:**
    *   Provide detailed information about the proposed changes.
    *   Address any comments and resolve them.
4.  **Validation Checks:** PRs are automatically checked for:
    *   Detection Template Structure (YAML).
    *   KQL Query Syntax.
5.  **Run Validations Locally:**
    *   **KQL Validation:** Requires .Net Core 3.1 SDK. Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and run `dotnet test`.
    *   **Detection Schema Validation:** Requires .Net Core 3.1 SDK. Navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and run `dotnet test`.
6.  **CLA-bot:**  A CLA-bot will guide you through the CLA process when you submit a PR.

Refer to the [wiki](https://aka.ms/threathunters) for further contribution details.

## Code of Conduct

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.