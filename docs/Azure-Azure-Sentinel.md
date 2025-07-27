# Azure Sentinel and Microsoft 365 Defender Security Content Repository

**Enhance your security posture with pre-built detections, hunting queries, and more for Microsoft Sentinel and Microsoft 365 Defender.**  [Explore the original repository](https://github.com/Azure/Azure-Sentinel).

This repository serves as a centralized resource for security professionals seeking to leverage the power of Microsoft Sentinel and Microsoft 365 Defender. It provides a wealth of pre-built content to help you secure your environment and proactively hunt for threats.

**Key Features:**

*   **Out-of-the-Box Detections:** Implement immediate threat detection with pre-configured rules.
*   **Exploration and Hunting Queries:**  Uncover hidden threats with advanced hunting capabilities.
*   **Workbooks:**  Visualize your security data and gain valuable insights.
*   **Playbooks:** Automate security responses and streamline your workflow.
*   **Microsoft 365 Defender Integration:** Includes hunting queries optimized for advanced hunting within both Microsoft 365 Defender and Microsoft Sentinel.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

**Feedback & Support:**

We value your input and are here to help.  Connect with the community and provide feedback through the following channels:

1.  **SIEM and SOAR Q&A:** [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Bug Reports:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

This project welcomes contributions and suggestions.  See the [wiki](https://aka.ms/threathunters) for more information.

### Contribution License Agreement (CLA)
Most contributions require a CLA. You must agree to a Contributor License Agreement (CLA) declaring that you have the right to grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

**General Steps for Contributing:**

1.  [Fork the repository](https://docs.github.com/github/getting-started-with-github/fork-a-repo)
2.  [Clone the repository](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
3.  [Create a branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work) for your changes.
4.  Make your changes and commit them to your branch.
5.  [Push your changes](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository).
6.  Submit a [Pull Request (PR)](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).  Provide detailed descriptions of your changes.

### Pull Request Validation Checks
Automated checks run on submitted pull requests:

*   **Structure Validation:** Ensures YAML files meet the required structure, particularly for Detections.
*   **KQL Validation:** Validates the syntax of KQL queries within templates.
*   **Detection Schema Validation:** Validates the schema of a detection, ensuring correct formats and attribute usage.

For more information on these validation checks, how to run the checks locally, and more, see the [wiki](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.