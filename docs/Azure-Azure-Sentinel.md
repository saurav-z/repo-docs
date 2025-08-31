# Microsoft Sentinel and Microsoft 365 Defender: Your One-Stop Shop for Security Content

**Enhance your threat detection and response capabilities with the unified Microsoft Sentinel and Microsoft 365 Defender repository.** This repository provides a wealth of resources to help you secure your environment and hunt for threats effectively.  Find everything you need, from out-of-the-box detections to hunting queries, workbooks, and playbooks, all in one place.  [Explore the original repository here](https://github.com/Azure/Azure-Sentinel).

**Key Features:**

*   **Pre-built Security Content:** Access ready-to-use detections, exploration queries, hunting queries, workbooks, and playbooks.
*   **Unified Security:** Includes content for both Microsoft Sentinel and Microsoft 365 Defender, including advanced hunting scenarios.
*   **Community Driven:**  Contribute your own security content and collaborate with the community.
*   **Advanced Hunting:** Includes Microsoft 365 Defender hunting queries for advanced threat hunting.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

**Get Involved and Provide Feedback:**

We value your input! Connect with us and share your feedback through the following channels:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for general product-specific Q&A.
2.  **XDR Q&A:** Participate in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for product-specific Q&A.
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:**  Use the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) for feedback on the community and contribution process.

**Contribution Guidelines:**

This project welcomes contributions! Please review the following guidelines:

*   **Contributor License Agreement (CLA):**  You must agree to a Contributor License Agreement (CLA).  Learn more at [https://cla.microsoft.com](https://cla.microsoft.com).
*   **Contributing to GitHub:**
    *   If you are a first-time contributor, start by [forking the repo](https://docs.github.com/github/getting-started-with-github/fork-a-repo).
    *   [Clone the repo](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository).
    *   [Create your own branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work).
    *   Make your additions/updates.
    *   Be sure to merge master back to your branch before you push.
    *   [Push your changes to GitHub](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository)
*   **Pull Requests:**
    *   Submit a [Pull Request (PR)](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).
    *   Provide detailed descriptions of your proposed changes.
    *   Address any comments and resolve them.
*   **PR Checks:** Ensure your Pull Request passes the structure and KQL validation tests by checking for any errors. The detection schema validation includes the detection's frequency and period, the detection's trigger type and threshold, validity of connectors Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)).
    *   **Run KQL Validation Locally:** Before submitting a PR, you can run KQL validations locally. You will need to have **.Net Core 3.1 SDK** installed to do so.  Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and execute `dotnet test`.  See the original documentation for examples of the expected output.
    *   **Run Detection Schema Validation Locally:** You can run detection schema validations locally.  You will need to have **.Net Core 3.1 SDK** installed to do so.  Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and execute `dotnet test`.

*   A CLA-bot will automatically determine whether you need to provide
    a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
    provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

For detailed information on contributing and available content, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).