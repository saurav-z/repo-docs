# Microsoft Sentinel & Microsoft 365 Defender Security Content Repository

Enhance your security posture and proactively hunt for threats with this comprehensive repository of detections, queries, and playbooks for Microsoft Sentinel and Microsoft 365 Defender.  [Visit the original repository](https://github.com/Azure/Azure-Sentinel) for the latest updates and contributions.

## Key Features:

*   **Out-of-the-Box Detections:**  Pre-built detections to identify and respond to threats quickly.
*   **Hunting Queries:**  Advanced queries for proactive threat hunting across your environment.
*   **Microsoft 365 Defender Integration:** Includes hunting queries for advanced threat hunting scenarios in both Microsoft 365 Defender and Microsoft Sentinel.
*   **Workbooks & Playbooks:**  Pre-built workbooks for data visualization and playbooks to automate security tasks.
*   **Community Driven:**  Benefit from community contributions and share your own security content.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved:  Contribute and Collaborate!

We welcome your contributions and feedback to improve this repository.

### Feedback and Support:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For contribution details, refer to the project's [wiki](https://aka.ms/threathunters).

### Contributing to the Repository:

1.  **Prerequisites:** Before contributing, make sure you are familiar with the [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) and, for specific steps for the Sentinel repo, refer to the [GettingStarted.md](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
2.  **Fork and Clone:** Fork the repository, clone it locally, and create a branch for your changes.
3.  **Make Changes:**  Use GitHub Desktop, Visual Studio, or VSCode to add or update content.
4.  **Merge and Push:** Merge your branch back with master before pushing your changes.
5.  **Submit a Pull Request (PR):**  Submit a pull request with detailed explanations of your changes.
6.  **PR Validation:** Ensure that your PR passes the Pull Request Detection Template Structure Validation Check and KQL Validation Check.
7.  **Review and Iterate:**  Address any comments and update your branch until the PR is approved.

### Pull Request Validation Details:

*   **Detection Template Structure Validation:** Ensures that the YAML structure of detection templates is correct.
*   **KQL Validation:** Validates the syntax of KQL queries. If custom logs are being used, ensure the table schema is defined in the appropriate JSON file.
*   **Run KQL Validation Locally:** To run the KQL validation locally, .Net Core 3.1 SDK needs to be installed. The command to execute is `dotnet test` inside the folder `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`.
*   **Detection Schema Validation:** Verifies attributes such as detection frequency, trigger types, and the validity of connector IDs.
*   **Run Detection Schema Validation Locally:**  To run the schema validation locally, .Net Core 3.1 SDK needs to be installed. The command to execute is `dotnet test` inside the folder `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`.

### Additional Information:

*   **CLA:** Contributions require agreeing to a Contributor License Agreement (CLA).
*   **Wiki:** For more information on contributing and getting started, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).