# Microsoft Sentinel and Microsoft 365 Defender: Your One-Stop Shop for Security Content

**Enhance your security posture and proactively hunt for threats with this comprehensive repository for Microsoft Sentinel and Microsoft 365 Defender.** This repository provides out-of-the-box detections, exploration queries, hunting queries, workbooks, and playbooks to help you secure your environment.

[Go to the original repository](https://github.com/Azure/Azure-Sentinel)

## Key Features:

*   **Pre-built Security Content:** Access a library of detections, queries, workbooks, and playbooks ready to deploy in your Microsoft Sentinel and Microsoft 365 Defender environments.
*   **Unified Security:** Includes hunting queries that work seamlessly across both Microsoft 365 Defender and Microsoft Sentinel for comprehensive threat detection.
*   **Community Driven:** Contribute to the repository and suggest new samples and resources.
*   **Advanced Hunting:** Utilize Microsoft 365 Defender hunting queries for in-depth threat analysis.
*   **Regular Updates:** Benefit from ongoing updates and new content designed to address evolving threat landscapes.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved:

We value your feedback and contributions! Here's how you can participate:

1.  **Ask Questions (SIEM & SOAR):** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for general product Q&A.
2.  **Ask Questions (XDR):** Join the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for general product Q&A.
3.  **Suggest Features:** Upvote or post new feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) for feedback on the community and contribution process.

## Contribution Guidelines:

We welcome contributions! Please review the following guidelines:

*   **Contributor License Agreement (CLA):** All contributions require you to agree to a CLA. For details, visit https://cla.microsoft.com.
*   **Contribution Process:**
    1.  **Fork, Clone, and Branch:** Fork the repository, clone it to your local machine, and create a new branch for your changes.
    2.  **Make Changes:** Use GitHub Desktop, Visual Studio, or VS Code to add or update content.
    3.  **Merge and Push:** Merge your changes from the master branch and push them to your branch.
    4.  **Submit a Pull Request (PR):** Submit a PR with detailed descriptions of your changes.
    5.  **PR Checks:** Automated checks are run to validate the structure and KQL queries.
    6.  **Address Feedback:** Make any necessary changes based on review comments.

*   **Detection Template Validation:** All detection contributions must follow the required YAML structure.
*   **KQL Validation:** Ensure your KQL queries are valid using the provided validation tools.
    *   **Run KQL Validation Locally:** Execute `dotnet test` in `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` after installing the .Net Core 3.1 SDK.
*   **Detection Schema Validation:** Schema of detections is automatically validated.
    *   **Run Schema Validation Locally:** Execute `dotnet test` in `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` after installing the .Net Core 3.1 SDK.

*   **Pull Request Best Practices:** Include detailed explanations in your pull requests for the reasons behind the change and what's changed in the code.

*   **For further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).**

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.