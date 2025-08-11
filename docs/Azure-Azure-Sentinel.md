# Microsoft Sentinel and Microsoft 365 Defender: Enhance Your Security Posture

**Enhance your security with a unified repository of detections, queries, workbooks, and playbooks for Microsoft Sentinel and Microsoft 365 Defender.** This repository offers valuable resources to help you proactively secure your environment and hunt for threats. ([View on GitHub](https://github.com/Azure/Azure-Sentinel))

## Key Features

*   **Out-of-the-Box Detections:** Implement pre-built detections to quickly identify and respond to threats.
*   **Hunting Queries:** Leverage hunting queries for both Microsoft Sentinel and Microsoft 365 Defender, including advanced hunting scenarios, to proactively seek out malicious activity.
*   **Exploration Queries:** Explore data using pre-built queries.
*   **Workbooks:** Utilize pre-built workbooks to visualize and understand your security data.
*   **Playbooks:** Automate security tasks with pre-built playbooks.
*   **Community Driven:** Submit requests for new samples and resources to tailor the repository to your specific needs.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback

Your feedback is crucial to the continuous improvement of these resources. Please use the following channels to ask questions, give feedback, and report issues:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit and vote on new feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **Community & Contribution Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We welcome contributions! Please review the [wiki](https://aka.ms/threathunters) to get started.

### Contributing to GitHub

1.  **Fork the Repository:** [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo)
2.  **Clone the Repository:** [Cloning a Repository](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
3.  **Create a Branch:** [Creating a Branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work)
4.  **Make your additions/updates.**
5.  **Merge your changes.** Be sure to merge master back to your branch before you push.
6.  **Push your Changes:** [Pushing Changes](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository)
7.  **Submit a Pull Request:** [About Pull Requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
    *   Provide detailed information on the proposed changes in your PR.

### Pull Request Checks

*   **Detection Template Structure Validation:** Ensure that all required sections of the YAML structure are included.
*   **KQL Validation:** Verify the syntax of your KQL queries.
*   **Schema Validation:** The schema validation includes the detection's frequency and period, the detection's trigger type and threshold, validity of connectors Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)), etc.

### Running KQL and Schema Validation Locally

*   Install the .NET Core 3.1 SDK. [How to download .Net](https://dotnet.microsoft.com/download)
*   For KQL Validation:
    *   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and run `dotnet test`.
    *   For custom tables, define your table schema in `Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables`.
*   For Schema Validation:
    *   Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and run `dotnet test`.

### Contributor License Agreement (CLA)

All contributions require agreeing to a Contributor License Agreement (CLA). Follow the instructions from the CLA-bot.

### Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.