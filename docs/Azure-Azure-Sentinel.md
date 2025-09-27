# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

This repository offers a wealth of security content, including detections, queries, workbooks, and playbooks, to help you proactively secure your environment with Microsoft Sentinel and Microsoft 365 Defender.  [Visit the original repo](https://github.com/Azure/Azure-Sentinel) for more details.

**Key Features:**

*   **Out-of-the-Box Detections:** Implement pre-built detections to quickly identify and respond to threats.
*   **Exploration and Hunting Queries:**  Uncover hidden threats with powerful hunting queries for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Comprehensive Security Content:**  Access workbooks, playbooks, and other resources to enhance your security posture.
*   **Microsoft 365 Defender Integration:** Leverage advanced hunting capabilities within Microsoft 365 Defender.
*   **Community Driven:**  Contribute and access community-created content to expand your security knowledge and capabilities.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

**Get Involved and Provide Feedback:**

Your input is valuable!  Use the following channels to ask questions, provide feedback, and report issues:

1.  **SIEM/SOAR Q&A:** [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Bug Reports:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **Community/Contribution Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

**Contribution Guidelines:**

This project welcomes contributions! Please review the following guidelines:

*   **Contributor License Agreement (CLA):**  You must agree to a CLA to contribute.  See [https://cla.microsoft.com](https://cla.microsoft.com) for details.
*   **Contribution Process:** Follow these steps to submit your contributions:
    1.  **Fork and Clone:** Fork the repository and clone it to your local machine.
    2.  **Create a Branch:** Create a new branch for your changes.
    3.  **Make Changes:**  Add or update your content.
    4.  **Merge and Push:** Merge your branch with the master and push your changes to GitHub.
    5.  **Submit Pull Request:**  Submit a pull request for review, providing details about your changes.

*   **Pull Request Checks:**
    *   **Detection Template Structure Validation:** Ensure your YAML files adhere to the required structure.
    *   **KQL Validation:**  Validate KQL queries for syntax errors.
    *   **Detection Schema Validation:** Ensure the schema of the detection is correct.
        *   **Run KQL Validation Locally:**
            1.  Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download).
            2.  Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` in your shell.
            3.  Run `dotnet test`.
        *   **Run Detection Schema Validation Locally:**
            1.  Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download).
            2.  Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` in your shell.
            3.  Run `dotnet test`.

*   **Contribution details:** Refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

*   **Code of Conduct:** This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.