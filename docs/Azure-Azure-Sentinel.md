# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender 

**This repository provides a wealth of security content, including detections, hunting queries, and playbooks, designed to help you effectively secure your environment using Microsoft Sentinel and Microsoft 365 Defender. [Visit the original repo](https://github.com/Azure/Azure-Sentinel).**

## Key Features

*   **Out-of-the-Box Detections:** Utilize pre-built detections to proactively identify and respond to threats.
*   **Hunting Queries:** Leverage powerful hunting queries, including Microsoft 365 Defender advanced hunting queries, to proactively search for threats.
*   **Workbooks:** Visualize and analyze security data using pre-configured workbooks.
*   **Playbooks:** Automate your security response with integrated playbooks.
*   **Unified Security Content:** Access a unified set of resources for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Community Driven:** Contribute and collaborate with the community to improve security content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback

We value your feedback and contributions. Here's how you can connect with the community:

1.  **General SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **General XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Product Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community & Contribution Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project welcomes contributions. Please review the following guidelines before submitting:

*   **Contributor License Agreement (CLA):**  All contributions require a CLA. Visit [https://cla.microsoft.com](https://cla.microsoft.com) for details.
*   **Contribution Process:**

    1.  **Fork and Clone:** If you're new, review the [GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) and clone the repository.
    2.  **Create a Branch:** Create a new branch for your changes.
    3.  **Make Changes:** Add or update content using your preferred method.
    4.  **Merge and Push:** Merge master to your branch and push your changes.
    5.  **Submit a Pull Request:** Submit a PR, including details about your changes.
    6.  **Review and Address Feedback:** Check the Pull Request and make any necessary changes.
*   **Pull Request Checks:**
    *   **Detection Template Structure Validation:**  Ensures all required parts of the YAML structure are included for detections.
    *   **KQL Validation Check:** Validates the syntax of KQL queries. If failures happen, Azure Pipeline can be used for debugging.
    *   **Detection Schema Validation Tests:** Checks frequency, trigger type, threshold, and connector IDs for correctness.
*   **Run Validations Locally:** You can run the KQL and Detection Schema validation checks locally using the instructions in the original README.

For further details on contributing and specific contribution types, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or comments.