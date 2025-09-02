# Enhance Your Security Posture with Microsoft Sentinel & Microsoft 365 Defender

**[Microsoft Sentinel and Microsoft 365 Defender](https://github.com/Azure/Azure-Sentinel) is your one-stop resource for pre-built security content to detect threats and secure your environment.**

This repository offers a wealth of resources to help you get started with Microsoft Sentinel and Microsoft 365 Defender.  It's designed to empower security professionals with the tools they need to proactively defend against cyber threats.

**Key Features:**

*   **Out-of-the-Box Detections:** Ready-to-use detections to identify potential threats in your environment.
*   **Hunting Queries:** Advanced hunting queries for both Microsoft Sentinel and Microsoft 365 Defender, allowing you to proactively search for threats.
*   **Workbooks:** Interactive workbooks to visualize and analyze your security data.
*   **Playbooks:** Automation playbooks to streamline your incident response.
*   **Microsoft 365 Defender Integration:** Enhanced hunting queries for advanced hunting scenarios in both Microsoft 365 Defender and Microsoft Sentinel.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)
*   [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
*   [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
*   [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)

**Contribute to the Community**

We encourage contributions! Review the [Contribution Guidelines](https://github.com/Azure/Azure-Sentinel/blob/master/README.md#contribution-guidelines) to learn how to submit your detections, queries, playbooks, and more.  For more information on what you can contribute and further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters). 

**Feedback & Support:**

*   **Report Bugs:**  File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
*   **Feature Requests:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)
*   **General Feedback:**  Submit a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)
*   **Questions:** Contact [AzureSentinel@microsoft.com](mailto:AzureSentinel@microsoft.com)

**Contribution Guidelines**

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA). For details, visit https://cla.microsoft.com.

**Note:** If you are a first-time contributor to this repository, follow [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md) before cloning or.

**General Steps to Add or Update Contributions:**

*   **Submit directly on GitHub website:**
    *   Browse to the folder you want to upload your file to.
    *   Choose "Upload Files" and browse to your file.
    *   You will be required to create your own branch and then submit the Pull Request for review.
*   **Use GitHub Desktop, Visual Studio, or VSCode:**
    *   [Fork the repo](https://docs.github.com/github/getting-started-with-github/fork-a-repo).
    *   [Clone the repo](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository).
    *   [Create your own branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work).
    *   Make your additions/updates.
    *   Be sure to merge master back to your branch before pushing.
    *   [Push your changes to GitHub](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository).

**Pull Request (PR) Process:**

*   Submit a [Pull Request (PR)](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) after pushing your changes.
*   Provide detailed information about the proposed changes.
*   Check the [Pull Request](https://github.com/Azure/Azure-Sentinel/pulls) for comments.
*   Make changes as suggested and update your branch or explain why no change is needed. Resolve the comment when done.

**PR Checks - Detection Template Structure and KQL Validation:**

*   **Structure Validation:** PR checks include structure validation to ensure required YAML structure parts are included.
*   **KQL Validation:** PR checks also run KQL query syntax validation. If it fails, use the Azure Pipeline link on the checks tab to diagnose the cause.

**Run KQL Validation Locally:**

*   Install .Net Core 3.1 SDK.
*   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` in the shell.
*   Run `dotnet test`.

**Detection schema validation tests**
The schema validation includes the detection's frequency and period, the detection's trigger type and threshold, validity of connectors Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)), etc.
A wrong format or missing attributes will result with an informative check failure, which should guide you through the resolution of the issue, but make sure to look into the format of already approved detection.

**Run Detection Schema Validation Locally:**

*   Install .Net Core 3.1 SDK.
*   Navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` in the shell.
*   Execute `dotnet test`

**CLA Requirements**

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and will guide you.

**Code of Conduct**

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions or comments.