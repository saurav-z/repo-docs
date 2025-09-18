# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

**Secure your environment and proactively hunt for threats with the unified Microsoft Sentinel and Microsoft 365 Defender repository.** This repository provides a wealth of security content, including out-of-the-box detections, exploration queries, hunting queries, workbooks, and playbooks to bolster your security operations.  

[Link to Original Repo:  https://github.com/Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel)

## Key Features:

*   **Pre-built Detections:** Leverage ready-to-use detection rules to quickly identify potential security threats.
*   **Hunting Queries:**  Uncover hidden threats with advanced hunting queries tailored for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Workbooks and Playbooks:** Streamline your security workflows with pre-built workbooks for data visualization and playbooks for automated incident response.
*   **Unified Security:**  Benefit from integrated security content that works seamlessly across Microsoft Sentinel and Microsoft 365 Defender.
*   **Community-Driven:** Contribute to and benefit from a community-driven repository, with new content regularly added.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved: Feedback & Contribution

We value your contributions and feedback.  Here's how to connect:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:**  Submit feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Bug Reports:** Report bugs via a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **Community & Contribution Feedback:**  Provide general feedback using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines:

This project welcomes contributions. Before contributing, please review the following:

*   **Contributor License Agreement (CLA):**  Ensure you agree to the CLA by visiting [https://cla.microsoft.com](https://cla.microsoft.com).
*   **Getting Started:** Refer to the [wiki](https://aka.ms/threathunters) for detailed contribution guidance.
*   **Contribution Process:**
    1.  **Fork and Clone:** Fork the repository and clone it to your local machine.
    2.  **Create a Branch:** Create a new branch for your changes.
    3.  **Add/Update Content:** Add or update your contributions.  
    4.  **Push Changes:** Push your changes to your branch on GitHub.
    5.  **Submit a Pull Request:**  Submit a Pull Request (PR) with a clear description of your changes, including details about the proposed change and why it is related to the code.

*   **PR Validation:**
    *   PRs undergo automated checks, including structure validation, KQL validation (syntax validation of KQL queries), and schema validation.
    *   For failing checks, refer to the Azure Pipeline for details.
    *   If needed, validate your table schema by referencing the `tablexyz.json` file for custom logs.

*   **Local Validation:** Run validation checks locally:
    *   **KQL Validation:** Execute `dotnet test` in the `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` directory (requires .Net Core 3.1 SDK).
    *   **Detection Schema Validation:** Execute `dotnet test` in the `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` directory (requires .Net Core 3.1 SDK).

*   **CLA-Bot:** A CLA-bot will automatically determine if you need to provide a CLA.

*   **Code of Conduct:**  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

*   **Get Started with the Wiki:** For contribution specifics, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).