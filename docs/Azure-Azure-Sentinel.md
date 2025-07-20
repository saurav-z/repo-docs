# Microsoft Sentinel and Microsoft 365 Defender: Your Central Hub for Security Content

This repository is your one-stop shop for pre-built detections, exploration queries, hunting queries, workbooks, playbooks, and more, designed to help you quickly ramp up with Microsoft Sentinel and enhance your security posture. [Explore the Azure Sentinel Repository](https://github.com/Azure/Azure-Sentinel).

## Key Features

*   **Unified Content:** Access a wide range of out-of-the-box detections, queries, workbooks, and playbooks for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Advanced Hunting:** Includes Microsoft 365 Defender hunting queries for in-depth threat hunting across your environment.
*   **Community Driven:** Contribute your own security content and benefit from contributions from the security community.
*   **Comprehensive Resources:** Links to vital documentation and community resources to help you get started quickly.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved

We value your feedback and contributions. Here's how you can get involved:

1.  **Q&A for SIEM and SOAR:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **Q&A for XDR:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit feature requests via the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We welcome contributions! Please review the following guidelines before submitting:

*   **Contributor License Agreement (CLA):**  Most contributions require you to agree to a CLA.  For details, visit https://cla.microsoft.com.
*   **Getting Started:** Refer to the [wiki](https://aka.ms/threathunters) for instructions.
*   **Adding New or Updated Contributions:**
    *   Submit directly on the GitHub website or use GitHub Desktop, Visual Studio, or VS Code.
    *   Fork and clone the repository.
    *   Create your own branch.
    *   Make your additions/updates.
    *   Push your changes to GitHub.
    *   Submit a Pull Request (PR).

### Pull Request Checks

*   **PR Details:**  Provide clear details about your proposed changes in the PR.
*   **Detection Template Structure Validation:**  Ensure your detection YAML structure is valid. See the contribution guidelines for more information.
*   **KQL Validation:**  KQL queries are syntax-validated.
*   **Schema Validation:** Ensure your detection schema is valid.
### Running Validations Locally:

To run validations locally:

*   You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
*   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\[Validation Name]\`
*   Execute `dotnet test`

### Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

For further information on what you can contribute and further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).