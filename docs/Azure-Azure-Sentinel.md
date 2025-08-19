# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

**Secure your environment and proactively hunt for threats with the comprehensive resources available in the Microsoft Sentinel and Microsoft 365 Defender repository.**  This repository offers a wealth of security content to help you get the most out of these powerful security tools.

[Go to the original repo](https://github.com/Azure/Azure-Sentinel)

**Key Features:**

*   **Pre-built Detections:** Discover out-of-the-box detections designed to identify potential threats.
*   **Hunting Queries:** Utilize advanced hunting queries, including Microsoft 365 Defender hunting queries, to proactively search for malicious activity.
*   **Exploration Queries:** Gain insights into your environment with pre-built exploration queries.
*   **Workbooks and Playbooks:** Leverage ready-to-use workbooks for visualization and playbooks for automated incident response.
*   **Unified Security:** Seamlessly integrate content for both Microsoft Sentinel (SIEM) and Microsoft 365 Defender (XDR).
*   **Community Driven:** Benefit from community contributions and resources.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved: Feedback and Support

We value your input! Here's how you can connect with us:

1.  **SIEM/SOAR Q&A:** Engage in the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community Feedback:** Use the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) for general feedback.

## Contribute to the Project

We welcome contributions and suggestions! Please review the following guidelines:

*   **Contributor License Agreement (CLA):** By contributing, you agree to the [Contributor License Agreement (CLA)](https://cla.microsoft.com), granting us the rights to use your contribution.
*   **Contribution Guidelines:**
    *   [General GitHub Fork guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) for first-time contributors.
    *   Follow the [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
    *   Use [GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop) or other IDE's to [Fork the repo](https://docs.github.com/github/getting-started-with-github/fork-a-repo), [Clone the repo](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository), and [Create your own branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work).
    *   After committing changes in your branch, be sure to merge master back to your branch before you push.
    *   [Push your changes to GitHub](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository) to submit your changes to the [Pull Request (PR)](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
*   **PR Requirements:**
    *   Include detailed explanations for all proposed changes to help speed up the review process.
    *   After submission, monitor the [Pull Request](https://github.com/Azure/Azure-Sentinel/pulls) for comments and address them as needed.
*   **PR Checks:**  All PRs will be checked for KQL and Schema validation
    *   **Detection Template Structure Validation:** The PR check runs a structure validation to make sure all required parts of the YAML structure are included.
    *   **KQL Validation:** The PR check runs a syntax validation of the KQL queries defined in the template.
    *   **Detection Schema Validation:** The schema validation includes the detection's frequency and period, the detection's trigger type and threshold, validity of connectors Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)), etc.
    *   **Run KQL and Detection Schema Validation Locally:**  Instructions are included in the original README to help with this.

For details on what you can contribute and further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.