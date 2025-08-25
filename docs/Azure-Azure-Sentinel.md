# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

**Boost your threat detection and incident response capabilities with pre-built security content for Microsoft Sentinel and Microsoft 365 Defender.** This repository provides a centralized hub for detections, queries, workbooks, and playbooks to help you proactively secure your environment. 

[Link to Original Repo: https://github.com/Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel)

## Key Features

*   **Pre-built Detections:** Leverage out-of-the-box detections to identify and respond to threats quickly.
*   **Hunting Queries:** Use powerful hunting queries for proactive threat hunting across Microsoft 365 Defender and Microsoft Sentinel.
*   **Workbooks & Playbooks:** Utilize pre-built workbooks for data visualization and playbooks to automate your security workflows.
*   **Unified Content:** Access a wide range of security content, including detections, queries, and playbooks, all in one place.
*   **Community Driven:** Contribute and collaborate with the community to improve and expand the repository's security content.
*   **Microsoft 365 Defender Integration:** Enhance your security posture with advanced hunting scenarios across Microsoft 365 Defender.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved: Feedback & Support

We value your feedback and encourage you to participate in the community.

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for general product questions.
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR specific product questions.
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community Feedback:** Use the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) for general feedback on the community and contribution process.

## Contributing Guidelines

We welcome contributions!  Please review the guidelines below to help contribute to the project:

*   **Contributor License Agreement (CLA):**  You must agree to a CLA before contributing. Visit https://cla.microsoft.com for details.

### How to Contribute:

1.  **Add Contributions via GitHub:**
    *   Upload files directly via the GitHub website (for smaller updates).
        *   Browse to the folder you want to upload your file to
        *   Choose Upload Files and browse to your file.
        *   You will be required to create your own branch and then submit the Pull Request for review.
    *   Use GitHub Desktop, Visual Studio, or VS Code for more complex changes.
        *   [Fork the repo](https://docs.github.com/github/getting-started-with-github/fork-a-repo)
        *   [Clone the repo](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
        *   [Create your own branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work)
        *   Make your changes in your local environment.
        *   Merge master back to your branch before you push.
        *   [Push your changes to GitHub](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository)

2.  **Submit a Pull Request (PR):**
    *   Provide detailed information about your changes in the PR.
    *   Address any comments from reviewers.
    *   Check the [Pull Request](https://github.com/Azure/Azure-Sentinel/pulls) for comments and update your branch as needed.

### PR Validation Checks

*   **Detection Template Structure Validation:** PRs are checked to ensure all required parts of the YAML structure are included, especially in detection templates.
*   **KQL Validation:** KQL queries in the template are validated for syntax.
    *   If your custom logs are using custom tables, ensure your table schema is in a json file in the *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables* folder.
    *   Run KQL validation locally:
        *   Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
        *   Open Shell and navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
        *   Run `dotnet test`
*   **Detection Schema Validation:** Validates the schema of a detection (frequency, period, trigger type, connector IDs, etc.).
    *   Run Detection Schema Validation locally:
        *   Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
        *   Open Shell and navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
        *   Run `dotnet test`

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

For more details, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).