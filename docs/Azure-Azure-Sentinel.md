# Microsoft Sentinel and Microsoft 365 Defender: Enhance Your Security Posture

**Fortify your cybersecurity defenses with ready-to-use detections, hunting queries, and more for Microsoft Sentinel and Microsoft 365 Defender!** This repository provides comprehensive security content to help you proactively identify and respond to threats, and is a collaborative space for security professionals.  Explore and contribute to a library of resources designed to enhance your security posture.

[Visit the original repository](https://github.com/Azure/Azure-Sentinel)

## Key Features:

*   **Out-of-the-box Detections:** Implement pre-built detection rules to identify potential threats.
*   **Hunting Queries:** Utilize hunting queries for both Microsoft Sentinel and Microsoft 365 Defender to proactively search for malicious activity.
*   **Workbooks & Playbooks:** Leverage pre-built workbooks for data visualization and playbooks for automated response.
*   **Community Contributions:** Benefit from community-driven content and contribute your own valuable resources.
*   **Unified Security:** Access content that spans Microsoft Sentinel and Microsoft 365 Defender for a comprehensive security approach.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback:

We value your input! Connect with the community and provide feedback through the following channels:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific discussions.
2.  **XDR Q&A:** Participate in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related inquiries.
3.  **Feature Requests:** Submit or upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** Report product or contribution bugs by filing a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community Feedback:** Provide general feedback on the community and contribution process by filing a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines:

This project welcomes contributions!  Review the [wiki](https://aka.ms/threathunters) to get started.

### How to Contribute:

1.  **Fork & Clone:** If new to this repository, [Fork the repo](https://docs.github.com/github/getting-started-with-github/fork-a-repo), and then [Clone the repo](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository).

2.  **Create a Branch:** [Create your own branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work).

3.  **Make Changes:** Add or update your contributions.

4.  **Merge Master:** Be sure to merge master back to your branch before you push.

5.  **Push Changes:** [Push your changes to GitHub](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository).

6.  **Submit a Pull Request:** Submit a [Pull Request (PR)](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) with a clear description of your changes.

7.  **Review PR Comments:** After submission, check the [Pull Request](https://github.com/Azure/Azure-Sentinel/pulls) for comments and make updates as needed.

### Pull Request Checks:

*   **Detection Template Structure Validation:**  Ensure that your detection YAML files adhere to the required structure.
*   **KQL Validation:** The syntax of KQL queries defined in the template will be validated.

    *   If the check fails, go to Azure Pipeline (by pressing on the errors link on the checks tab in your PR)

    *   If you are using custom logs table (a table which is not defined on all workspaces by default) you should verify your table schema is defined in json file in the folder *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables*
*   **Run KQL Validation Locally:**
    *   You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
    *   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Execute `dotnet test`
*   **Detection Schema Validation:** Ensure consistency and accuracy.

    *   A wrong format or missing attributes will result with an informative check failure, which should guide you through the resolution of the issue, but make sure to look into the format of already approved detection.
*   **Run Detection Schema Validation Locally:**
    *   You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
    *   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Execute `dotnet test`

### Contributor License Agreement (CLA):

All contributions require you to agree to a Contributor License Agreement (CLA). Follow the instructions provided by the CLA-bot.

### Code of Conduct:

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For detailed information on contributing, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).