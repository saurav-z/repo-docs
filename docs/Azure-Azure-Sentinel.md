# Microsoft Sentinel and Microsoft 365 Defender: Your Unified Security Hub

**Secure your environment and proactively hunt for threats with the comprehensive resources available in this repository, including detections, queries, workbooks, and playbooks.**  Find the original repository [here](https://github.com/Azure/Azure-Sentinel).

This repository provides a wealth of security content to help you get started with Microsoft Sentinel and Microsoft 365 Defender, and it welcomes contributions from the community.

## Key Features:

*   **Out-of-the-Box Detections:** Implement pre-built detections to identify and respond to threats.
*   **Exploration and Hunting Queries:** Utilize queries to investigate security incidents and proactively hunt for malicious activity.
*   **Workbooks:** Leverage pre-configured workbooks for data visualization and analysis.
*   **Playbooks:** Automate security responses and incident handling with pre-built playbooks.
*   **Microsoft 365 Defender Integration:** Includes hunting queries for advanced threat hunting in both Microsoft 365 Defender and Microsoft Sentinel.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support:

We value your feedback and are here to help.  Connect with the community and Microsoft for assistance:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific Q&A.
2.  **XDR Q&A:** Participate in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related discussions.
3.  **Feature Requests:** Submit and vote on feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Bug Reports:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Use the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) to provide feedback on the community and contribution process.

## Contribution Guidelines:

This project welcomes contributions. Follow these guidelines to contribute:

1.  **Contributor License Agreement (CLA):**  By contributing, you agree to a CLA.  See https://cla.microsoft.com for details.
2.  **Fork and Clone:**  Fork the repository and clone it to your local machine.  See [GettingStarted.md](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md) for specific steps.
3.  **Create a Branch:** Create a new branch for your contributions.
4.  **Make Changes:** Add or update files using GitHub Desktop, Visual Studio, or VS Code.
5.  **Merge and Push:** Merge your branch with the master and push your changes to your forked repository.
6.  **Submit a Pull Request (PR):**  Submit a PR with a detailed description of your changes.

**Pull Request Checks:**

*   **Detection Template Structure Validation:**  Ensure your YAML files include the required structure.
*   **KQL Validation:**  Verify the syntax of your KQL queries.
*   **Schema Validation:** Validate detection schema.

    *  Run KQL validation locally : `dotnet test`

*  Run Detection Schema Validation Locally:  `dotnet test`

    *   (Requires .NET Core 3.1 SDK)
   

For detailed contribution information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

## Code of Conduct:

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.