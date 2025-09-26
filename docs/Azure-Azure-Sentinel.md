# Azure Sentinel & Microsoft 365 Defender: Your Central Hub for Security Content and Threat Hunting

**Enhance your security posture and proactively hunt for threats with the unified Microsoft Sentinel and Microsoft 365 Defender repository.** This repository provides a wealth of resources, including detections, exploration queries, hunting queries, workbooks, and playbooks, all designed to help you get the most out of Microsoft Sentinel and Microsoft 365 Defender. Access pre-built security content and contribute to the community!

[![GitHub Repo stars](https://img.shields.io/github/stars/Azure/Azure-Sentinel?style=social)](https://github.com/Azure/Azure-Sentinel)

**Key Features:**

*   **Out-of-the-box Detections:** Implement pre-built detection rules to identify and respond to threats quickly.
*   **Hunting Queries:** Proactively search for malicious activity using pre-defined hunting queries.
*   **Exploration Queries:** Gain insights into your data and understand your environment.
*   **Workbooks:** Visualize your security data and gain valuable insights.
*   **Playbooks:** Automate your security tasks with pre-built playbooks.
*   **Microsoft 365 Defender Integration:** Includes advanced hunting queries for Microsoft 365 Defender.
*   **Community Driven:** Contribute your own detections, queries, and playbooks to help other security professionals.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved

We value your feedback and contributions! Here's how you can connect with the community:

1.  **Q&A for SIEM and SOAR:** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **Q&A for XDR:** Join the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Upvote or post new on [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We welcome contributions to this repository. To contribute, you must agree to the Contributor License Agreement (CLA). For details, visit https://cla.microsoft.com.

### How to Contribute

1.  **Fork the Repository:** If you're new to the repo, start by forking the repository.  Follow the [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) and/or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
2.  **Clone the Repository:** Clone the forked repository to your local machine.
3.  **Create a Branch:** Create a new branch for your changes.
4.  **Make Your Changes:** Add or update your contributions.
5.  **Push Your Changes:** Push your changes to your branch on your forked repository.
6.  **Submit a Pull Request:** Submit a pull request from your branch to the main repository.  Provide detailed information about your proposed changes.
7.  **Review and Feedback:** Address any comments or suggestions from the reviewers.

### Pull Request Validation Checks

Automated checks are run as part of the pull request process:

*   **Detection Template Structure Validation:** Ensures the YAML structure of detection templates is valid. If there are validation errors, see the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for required sections.
*   **KQL Validation:** Validates the syntax of KQL queries in detection templates. See example error messages and instructions for resolving errors in the original README.
*   **Detection Schema Validation:** Validates the detection's frequency and period, the detection's trigger type and threshold, etc.

#### Run Validation Locally:

*   **KQL Validation:** Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and run `dotnet test`.
*   **Detection Schema Validation:** Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and run `dotnet test`.

### Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or comments.

### Additional Resources
*   For information on what you can contribute and further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

**[Visit the original repository](https://github.com/Azure/Azure-Sentinel) to explore the latest security content and contribute to the community!**