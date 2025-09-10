# Supercharge Your Security with Microsoft Sentinel & Microsoft 365 Defender

**This repository is your one-stop shop for security content, providing out-of-the-box detections, hunting queries, playbooks, and more to elevate your Microsoft Sentinel and Microsoft 365 Defender security posture.**  Access all of the security content, with queries for both Microsoft Sentinel and Microsoft 365 Defender.

[Link to Original Repository](https://github.com/Azure/Azure-Sentinel)

## Key Features

*   **Pre-built Detections:** Get started quickly with pre-configured detections to identify potential threats.
*   **Hunting Queries:** Proactively search for threats using advanced hunting queries, including Microsoft 365 Defender queries for advanced hunting scenarios.
*   **Workbooks & Playbooks:** Visualize your data and automate your incident response with pre-built workbooks and playbooks.
*   **Unified Content:** Access security content that integrates with both Microsoft Sentinel and Microsoft 365 Defender.
*   **Community Driven:** Contribute to the community and leverage contributions from other security professionals.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback

We value your feedback and contributions! Here's how you can participate:

1.  **General Q&A for SIEM & SOAR:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for discussions.
2.  **General Q&A for XDR:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related discussions.
3.  **Feature Requests:** Suggest new features or upvote existing requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** Submit bug reports using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Feedback on Community & Contribution:** Use the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) to provide general feedback.

## Contribution Guidelines

This project welcomes contributions!  Please review and follow the contribution guidelines, including:

*   **Contributor License Agreement (CLA):** You must agree to the [CLA](https://cla.microsoft.com) to contribute.
*   **Contribution Process:** Use the below methods to contribute:
    *   Submit for review directly on GitHub website 
        * Browse to the folder you want to upload your file to
        * Choose Upload Files and browse to your file. 
        * You will be required to create your own branch and then submit the Pull Request for review.
    *   Use [GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop) or [Visual Studio](https://visualstudio.microsoft.com/vs/) or [VSCode](https://code.visualstudio.com/?wt.mc_id=DX_841432)
        * [Fork the repo](https://docs.github.com/github/getting-started-with-github/fork-a-repo)  
        * [Clone the repo](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
        * [Create your own branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work)
        * Do your additions/updates in GitHub Desktop
        * Be sure to merge master back to your branch before you push. 
        * [Push your changes to GitHub](https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository)

*   **Pull Requests (PRs):** Provide clear details in your PRs, including the reason for the change and its relationship to the code. After submission, check the [Pull Request](https://github.com/Azure/Azure-Sentinel/pulls) for comments and address them.
*   **PR Checks**:
    *   **Detection Template Structure Validation:** Ensure your YAML files meet the required structure for detections.
    *   **KQL Validation:** Verify that the KQL queries in your templates are valid, before submitting Pull Request in you local machine:
        * You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
        * Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
        * Execute `dotnet test`
    *   **Detection Schema Validation:** Ensure that the schema of the detection is valid, before submitting Pull Request in you local machine:
        * You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
        * Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
        * Execute `dotnet test`

*   **CLA-bot:**  A CLA-bot will guide you through the process if you need to provide a CLA.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

## Further Information

For more details on contributions and other information, check the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section of the project's [wiki](https://aka.ms/threathunters).