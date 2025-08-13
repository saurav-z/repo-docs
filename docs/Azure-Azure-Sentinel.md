# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

This repository offers a wealth of security content, including detections, hunting queries, workbooks, and playbooks to help you proactively defend your environment using **Microsoft Sentinel** and **Microsoft 365 Defender**. ([Original Repo](https://github.com/Azure/Azure-Sentinel))

## Key Features

*   **Out-of-the-Box Detections:** Pre-built rules to identify threats and security incidents.
*   **Hunting Queries:**  Enable proactive threat hunting across your environment.
*   **Workbooks:**  Visualize security data and gain insights.
*   **Playbooks:** Automate incident response and security tasks.
*   **Microsoft 365 Defender Integration:** Includes hunting queries to extend your security reach.
*   **Community Driven:**  Contribute and request new content to meet your specific needs.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Help and Provide Feedback

We value your feedback and encourage you to engage with the community through the following channels:

1.  **SIEM/SOAR Q&A:** [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribute to the Community

This project welcomes contributions from the community!  Please review the [wiki](https://aka.ms/threathunters) to get started.

### Contribution Guidelines

1.  **Contributor License Agreement (CLA):**  All contributions require agreement to a CLA.  Visit [https://cla.microsoft.com](https://cla.microsoft.com) for details.
2.  **Fork and Contribute:** Use GitHub to submit new or updated content
    *   Submit directly on the GitHub website.
    *   Use GitHub Desktop, Visual Studio, or VS Code.
        *   Fork the repo.
        *   Clone the repo.
        *   Create a branch.
        *   Make your changes.
        *   Merge master back to your branch before you push.
        *   Push your changes to GitHub.
        *   Submit a Pull Request (PR).
3.  **Pull Request (PR) Details:**  Provide clear details about your changes.
4.  **PR Checks:**  Automated checks validate YAML structure and KQL queries.

### Pull Request Validation Checks

*   **Detection Schema Validation:** Ensure your detection adheres to the required schema.
*   **KQL Validation:** Validate your KQL queries for syntax errors.
    *   To run the KQL validation locally:
        *   Install .Net Core 3.1 SDK.
        *   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
        *   Run `dotnet test`

## Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.