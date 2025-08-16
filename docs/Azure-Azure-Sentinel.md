# Microsoft Sentinel and Microsoft 365 Defender: Your Unified Security Hub

**Enhance your threat detection and response capabilities with this comprehensive repository for Microsoft Sentinel and Microsoft 365 Defender.** ([Original Repository](https://github.com/Azure/Azure-Sentinel))

This repository provides a wealth of resources to help you secure your environment, including:

*   **Out-of-the-box Detections:** Ready-to-use detection rules to identify threats.
*   **Exploration & Hunting Queries:** Pre-built queries for proactive threat hunting.
*   **Hunting Queries for Microsoft 365 Defender:** Advanced hunting capabilities within both Microsoft 365 Defender and Microsoft Sentinel.
*   **Workbooks:** Customizable dashboards for data visualization and analysis.
*   **Playbooks:** Automation workflows to streamline incident response.

## Key Features

*   **Unified Security:** Combine the power of Microsoft Sentinel and Microsoft 365 Defender.
*   **Community-Driven:** Leverage contributions from the security community.
*   **Extensive Content:** Access a wide range of detections, queries, and automation.
*   **Easy Onboarding:** Resources to quickly get started with Microsoft Sentinel.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback & Support

We value your input. Engage with the community through:

1.  **SIEM/SOAR Q&A:** [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We encourage contributions! Please review the following guidelines:

### How to Contribute

1.  **Fork and Clone:** Fork the repository and clone it to your local machine.
2.  **Create a Branch:** Create a new branch for your changes.
3.  **Make Changes:** Add or update content using your preferred method.
4.  **Merge and Push:** Merge changes from master into your branch before pushing to GitHub.
5.  **Submit a Pull Request:** Submit a pull request for review.

### Additional Resources

*   [General GitHub Fork guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo)
*   [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md)
*   [GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop)
*   [Visual Studio](https://visualstudio.microsoft.com/vs/)
*   [VSCode](https://code.visualstudio.com/?wt.mc_id=DX_841432)

### Pull Request Checks

*   **Detection Template Structure Validation:** Ensure your YAML files adhere to the required structure.
*   **KQL Validation:** Validate the syntax of your KQL queries. If you are using custom logs table (a table which is not defined on all workspaces by default) you should verify your table schema is defined in json file in the folder Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables
*   **Detection Schema Validation:** Test the schema of a detection.

#### Run KQL Validation Locally
1.  You need to have .Net Core 3.1 SDK installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
2.  Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
3.  Execute `dotnet test`
#### Run Detection Schema Validation Locally
1.  You need to have .Net Core 3.1 SDK installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
2.  Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
3.  Execute `dotnet test`

### Contributor License Agreement (CLA)

Contributions require a Contributor License Agreement (CLA). Follow the instructions provided by the CLA-bot when submitting a pull request.

### Code of Conduct

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For more information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).