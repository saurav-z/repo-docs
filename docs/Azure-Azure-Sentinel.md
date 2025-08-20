# Microsoft Sentinel & Microsoft 365 Defender: Your Unified Security Solution

**Enhance your security posture with pre-built detections, hunting queries, workbooks, and playbooks in this comprehensive repository for Microsoft Sentinel and Microsoft 365 Defender.** [(Original Repo)](https://github.com/Azure/Azure-Sentinel)

## Key Features:

*   **Pre-built Detections:** Get started quickly with out-of-the-box detections designed to identify threats in your environment.
*   **Hunting Queries:** Proactively search for threats with a collection of hunting queries, including advanced hunting scenarios in both Microsoft 365 Defender and Microsoft Sentinel.
*   **Workbooks & Playbooks:** Leverage pre-built workbooks for data visualization and playbooks for automated incident response.
*   **Unified Security:** Access content that spans both Microsoft Sentinel (SIEM) and Microsoft 365 Defender (XDR) for comprehensive threat protection.
*   **Community Driven:** Benefit from community contributions and share your own detections, queries, and more.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback:

We value your input! Here's how you can contribute and ask questions:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for general product discussions.
2.  **XDR Q&A:** Participate in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related discussions.
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Submit a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) for community or contribution feedback.

## Contribution Guidelines:

We welcome contributions! Please review the following guidelines before submitting:

*   **Contributor License Agreement (CLA):** By contributing, you agree to the terms of the CLA.  Visit [https://cla.microsoft.com](https://cla.microsoft.com) for details.
*   **Fork & Contribute:**  Follow these steps:
    1.  [Fork the repo](https://docs.github.com/github/getting-started-with-github/fork-a-repo).
    2.  [Clone the repo](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository).
    3.  [Create your own branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work).
    4.  Add/Update your content.
    5.  Merge master into your branch before pushing.
    6.  [Push your changes to GitHub](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository).
    7.  Submit a [Pull Request (PR)](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).

## Automated Checks During Pull Request

*   **Detection Template Structure Validation:** Ensure your YAML files adhere to the required structure; see the contribution guidelines for details.  A missing section will cause a validation error.
*   **KQL Validation:**  The queries are validated for syntax.

    *   If you use custom logs tables: make sure your table schema is defined in JSON file in the folder `Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables`

*   **Detection Schema Validation:** Checks for format or missing attributes, see existing detections to ensure accuracy.

## Run Validations Locally

To check the validations on your local machine, you need to have .Net Core 3.1 SDK.
### KQL Validation
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
* Execute `dotnet test`
### Detection Schema Validation
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
* Execute `dotnet test`

## Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.

For additional details and contribution ideas, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).