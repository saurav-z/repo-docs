# Enhance Your Security Posture with Microsoft Sentinel & Microsoft 365 Defender

**Unlock proactive threat detection and incident response with the Microsoft Sentinel and Microsoft 365 Defender repository, your one-stop resource for security content and best practices.** ([Original Repo](https://github.com/Azure/Azure-Sentinel))

This repository provides a wealth of resources to help you get started with Microsoft Sentinel and Microsoft 365 Defender.  It includes pre-built detections, hunting queries, workbooks, and playbooks to help secure your environment and proactively hunt for threats.

## Key Features:

*   **Out-of-the-Box Detections:**  Immediately identify and respond to threats with pre-configured detection rules.
*   **Hunting Queries:** Proactively search for malicious activity and indicators of compromise.
*   **Workbooks:**  Visualize security data and gain valuable insights through interactive dashboards.
*   **Playbooks:**  Automate incident response and streamline security operations.
*   **Microsoft 365 Defender Integration:** Leverage advanced hunting capabilities for comprehensive threat detection across your Microsoft 365 environment.
*   **Community Driven:** Contribute to and benefit from a community-sourced library of security content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved: Feedback & Support

Your feedback is invaluable!  Connect with the community and get your questions answered through these channels:

1.  **SIEM & SOAR Q&A:**  Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:**  Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit or upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:**  File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:**  File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contributing

We welcome contributions!  To get started, review the following guidelines:

### Contribution License Agreement (CLA)

By contributing, you agree to a Contributor License Agreement (CLA).  Learn more at [https://cla.microsoft.com](https://cla.microsoft.com).

### How to Contribute:

1.  **Fork and Clone:** Fork the repository and clone it to your local machine. ([General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo), [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md)).
2.  **Create a Branch:** Create a new branch for your changes.
3.  **Make Changes:**  Add or update content (detections, queries, workbooks, etc.).  Use GitHub Desktop, Visual Studio, or VS Code.
4.  **Merge and Push:** Merge the master branch back into your branch and push your changes to your fork.
5.  **Submit a Pull Request:**  Create a Pull Request (PR) on GitHub, providing detailed information about your changes.  Review the [Pull Request guidelines](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).

### Pull Request Validation Checks

Automated checks are performed on all Pull Requests:

*   **Detection Template Structure Validation:** Ensures that YAML files for detections adhere to the required structure.  Errors will be displayed if required sections are missing.
*   **KQL Validation:** Validates the syntax of KQL queries within the templates.  Errors will identify any syntax issues. To run the KQL validation locally, you will need to have .Net Core 3.1 SDK installed and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and run `dotnet test`.
*   **Detection Schema Validation:** Validates the schema of a detection to ensure all attributes are correctly formatted. You will need to have .Net Core 3.1 SDK installed and navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and run `dotnet test`.

### Additional Guidelines

*   Review the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for detailed information.
*   A CLA-bot will guide you through the CLA process when you submit a pull request.

For more details and information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  For questions, contact [opencode@microsoft.com](mailto:opencode@microsoft.com).