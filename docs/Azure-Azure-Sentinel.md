# Microsoft Sentinel and Microsoft 365 Defender: Security Content and Threat Hunting Resources

**Enhance your cybersecurity posture with out-of-the-box detections, hunting queries, and more using the comprehensive resources in this repository.** This repository, [Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel), provides a unified source for Microsoft Sentinel and Microsoft 365 Defender security content, empowering you to proactively defend against threats.

**Key Features:**

*   **Pre-built Detections:** Get started quickly with ready-to-use detections.
*   **Exploration and Hunting Queries:** Uncover threats with advanced hunting capabilities, including Microsoft 365 Defender queries.
*   **Workbooks and Playbooks:** Visualize data and automate incident response.
*   **Community-Driven:** Contribute and suggest improvements to enhance the security content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support

We value your input and provide multiple channels for questions, feedback, and feature requests:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit and upvote on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We welcome contributions!

### Contribution Process

1.  **Fork the Repository:** Begin by forking the repository. ([General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo)  or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md))
2.  **Create a Branch:** Create a new branch for your changes.
3.  **Make Changes:** Add or update content.
4.  **Submit a Pull Request:** Submit a pull request for review. Include a detailed description of your proposed changes.

### Important Checks

*   **Pull Request Detection Template Structure Validation:** Ensure your contributions adhere to the required YAML structure.
*   **Pull Request KQL Validation:** Verify the syntax of your KQL queries using the provided validation tools.  (See the [Running KQL Validation Locally](#run-kql-validation-locally))
*   **Detection Schema Validation Tests:** Your Detection schema will be automatically validated as part of the pull request.  (See the [Run Detection Schema Validation Locally](#run-detection-schema-validation-locally))

### Code of Conduct and CLA

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contributions require agreeing to a [Contributor License Agreement (CLA)](https://cla.microsoft.com).

For detailed contribution information, refer to the project's [wiki](https://aka.ms/threathunters).

### Run KQL Validation Locally
* You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
* Execute `dotnet test`

### Run Detection Schema Validation Locally
* You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
* Execute `dotnet test`