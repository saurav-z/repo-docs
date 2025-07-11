# Microsoft Sentinel and Microsoft 365 Defender Security Content Repository

**Enhance your security posture with pre-built detections, hunting queries, workbooks, and playbooks for Microsoft Sentinel and Microsoft 365 Defender.** This repository provides a comprehensive library of security content to help you secure your environment and proactively hunt for threats. [Explore the Azure Sentinel Repository](https://github.com/Azure/Azure-Sentinel) for a wealth of resources.

## Key Features

*   **Out-of-the-Box Detections:** Quickly identify potential threats with pre-configured detections.
*   **Hunting Queries:** Proactively search for malicious activity with tailored hunting queries, including Microsoft 365 Defender queries.
*   **Workbooks:** Visualize your security data and gain insights through interactive workbooks.
*   **Playbooks:** Automate your security operations with pre-built playbooks.
*   **Unified Content:** Access security content for both Microsoft Sentinel and Microsoft 365 Defender in one place.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support

We value your feedback!  Here's how to get in touch:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contributing

This project welcomes contributions! Review the contribution guidelines below for details on how to get involved:

### Contribution Guidelines

*   **Contribution License Agreement (CLA):** All contributions require agreeing to a CLA. More information is available at [https://cla.microsoft.com](https://cla.microsoft.com).
*   **Getting Started:** If you are a first-time contributor, refer to the [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) and  [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
*   **Contribution Methods:**

    *   **Direct GitHub Upload:** Upload files directly through the GitHub interface by browsing to the folder and using the "Upload Files" option. You will need to create a new branch and submit a Pull Request (PR).
    *   **GitHub Desktop, Visual Studio, or VS Code:** Fork, clone, and create a branch for your changes. Push the changes and submit a Pull Request (PR).
*   **Pull Request (PR) Process:**

    *   Include detailed explanations of your changes.
    *   Check the PR for comments after submission.
    *   Address any suggestions.
*   **Validation Checks:**  Pull Requests are automatically validated. If there are errors, check the Azure Pipeline (accessible via the errors link) for detailed information.

### Validation Checks Details

*   **Detection Template Structure Validation Check:** Ensure your YAML files have all required sections.
*   **KQL Validation Check:** Verify the syntax of your KQL queries. If you're using custom logs table (a table which is not defined on all workspaces by default) you should verify your table schema is defined in json file in the folder *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables*
*   **Run KQL Validation Locally**:
    *   Install **.Net Core 3.1 SDK** [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
    *   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Execute `dotnet test`
*   **Detection Schema Validation Tests:** Verify the schema of a detection.  A wrong format or missing attributes will result with an informative check failure.
*   **Run Detection Schema Validation Locally:**
    *   Install **.Net Core 3.1 SDK** [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
    *   Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Execute `dotnet test`

## Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

## Get Started

For more details on what you can contribute and further information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).