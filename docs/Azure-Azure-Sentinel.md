# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

Secure your environment and proactively hunt for threats with the unified resources in the Microsoft Sentinel and Microsoft 365 Defender repository, your one-stop shop for security content. [Go to the original repository](https://github.com/Azure/Azure-Sentinel)

## Key Features:

*   **Out-of-the-Box Detections:** Quickly implement pre-built detections to identify and respond to threats.
*   **Exploration and Hunting Queries:** Leverage pre-written KQL queries for proactive threat hunting and in-depth investigation.
*   **Playbooks and Workbooks:** Utilize automation through playbooks and interactive dashboards for enhanced security operations.
*   **Microsoft 365 Defender Integration:** Access hunting queries specifically designed for advanced hunting scenarios across Microsoft 365 Defender and Microsoft Sentinel.
*   **Community Driven:** Contribute and request content to enhance the collective security knowledge base.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support:

We value your feedback and encourage you to participate in our community.

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit your ideas on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:**  Report product or contribution bugs via a [GitHub Issue](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=) using the Bug template.
5.  **Community & Contribution Feedback:** Provide general feedback via a [GitHub Issue](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) using the Feature Request template.

## Contribution Guidelines:

This project thrives on community contributions.  Before contributing, please review the [wiki](https://aka.ms/threathunters) to get started.

1.  **Contributor License Agreement (CLA):**  All contributions require agreement to the Contributor License Agreement (CLA).  Visit https://cla.microsoft.com for details.
2.  **Adding Contributions:**

    *   **Directly on GitHub:** Upload files directly via the GitHub interface.
    *   **GitHub Desktop / Visual Studio / VSCode:**
        *   Fork the repository.
        *   Clone the repository.
        *   Create a new branch.
        *   Make your changes.
        *   Merge master back to your branch before pushing.
        *   Push your changes.
3.  **Pull Requests (PR):**

    *   Submit a PR with detailed descriptions of the changes.
    *   Address any comments from reviewers.
    *   Check the [Pull Requests](https://github.com/Azure/Azure-Sentinel/pulls)
4.  **PR Checks:** Automatic validation includes structure and KQL (query) checks:

    *   **Detection Template Structure Validation:** Ensure all required sections are present in your YAML files. Errors will highlight missing elements.
    *   **KQL Validation Check:** Verify that your KQL queries are syntactically correct.  See example errors in the documentation.
        *   For custom logs, verify your table schema is defined in a JSON file in the appropriate directory.
    *   **Run KQL Validation Locally:**  Use `dotnet test` in the `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` directory (requires .NET Core 3.1 SDK).
    *   **Detection Schema Validation Tests:** Validates frequency, trigger types, and connector IDs.
    *   **Run Detection Schema Validation Locally:**  Use `dotnet test` in the `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` directory (requires .NET Core 3.1 SDK).

5.  **CLA-bot:**  The CLA-bot will guide you through the CLA process.

## Code of Conduct:

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.