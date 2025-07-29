# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

This repository is your central hub for security content, offering out-of-the-box detections, hunting queries, workbooks, and playbooks to help you get started with Microsoft Sentinel and enhance your threat hunting capabilities.

[Go to the Azure Sentinel Repository](https://github.com/Azure/Azure-Sentinel)

## Key Features:

*   **Pre-built Detections:** Leverage ready-to-use detections to identify and respond to threats.
*   **Hunting Queries:** Utilize advanced hunting queries, including those for Microsoft 365 Defender, to proactively search for threats.
*   **Comprehensive Resources:** Access a wealth of resources, including workbooks and playbooks, to streamline your security operations.
*   **Community Driven:** Contribute to and benefit from a community-driven repository of security content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support

We value your feedback and encourage you to engage with the community. Here's how to get in touch:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific discussions.
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related inquiries.
3.  **Feature Requests:** Submit feature requests or vote on existing ones in the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Provide feedback on the community and contribution process via a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

We welcome contributions! To contribute, please review our [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries) and adhere to the following steps:

### General Steps for Contribution
*   Submit for review directly on GitHub website 
*   Use [GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop) or [Visual Studio](https://visualstudio.microsoft.com/vs/) or [VSCode](https://code.visualstudio.com/?wt.mc_id=DX_841432)
    *   [Fork the repo](https://docs.github.com/github/getting-started-with-github/fork-a-repo)  
    *   [Clone the repo](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
    *   [Create your own branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work)
    *   Do your additions/updates in GitHub Desktop
    *   Be sure to merge master back to your branch before you push. 
    *   [Push your changes to GitHub](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository)

### Pull Request
*   Submit a [Pull Request (PR)](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
*   Include details about the proposed changes.
*   Review and respond to comments on the PR.

#### Detection Template Structure Validation Check
As part of the PR checks a structure validation is run to make sure all required parts of the YAML structure are included.

#### Pull Request KQL Validation Check
Syntax validation of the KQL queries defined in the template is performed.

#### Run KQL Validation Locally
*   Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
*   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
*   Execute `dotnet test`

#### Detection schema validation tests
There is an automatic validation of the schema of a detection.
A wrong format or missing attributes will result with an informative check failure.

#### Run Detection Schema Validation Locally
*   Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
*   Navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
*   Execute `dotnet test`

### Contributor License Agreement (CLA)

All contributions require a Contributor License Agreement (CLA). Follow the instructions provided by the CLA-bot when submitting a pull request.

## Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).

For further details on contributing, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).