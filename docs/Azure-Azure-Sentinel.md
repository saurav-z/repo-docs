# Microsoft Sentinel and Microsoft 365 Defender: Security Content Repository

**Enhance your security posture with pre-built detections, hunting queries, and more for Microsoft Sentinel and Microsoft 365 Defender, all in one place!** ([Original Repository](https://github.com/Azure/Azure-Sentinel))

This repository is your go-to resource for security professionals looking to leverage the power of Microsoft Sentinel and Microsoft 365 Defender. It provides a wealth of resources to help you secure your environment and proactively hunt for threats.

## Key Features:

*   **Out-of-the-box Detections:** Implement pre-configured detections to identify and respond to threats quickly.
*   **Exploration and Hunting Queries:** Utilize a library of queries for in-depth investigation and threat hunting.
*   **Workbooks and Playbooks:** Streamline your security operations with pre-built workbooks for data visualization and playbooks for automated responses.
*   **Microsoft 365 Defender Integration:** Includes hunting queries specifically designed for advanced hunting scenarios within Microsoft 365 Defender.
*   **Community Driven:**  Contribute your own security content and help build a more comprehensive resource for everyone.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support:

We value your input and encourage you to share your questions and feedback through the following channels:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for general product-specific questions.
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related inquiries.
3.  **Feature Requests:** Suggest new features or upvote existing ones on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Submit general feedback using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contributing Guidelines:

This project welcomes contributions!  Please review the [wiki](https://aka.ms/threathunters) for contribution details.

### Contribution Process:

1.  **Fork the repository** if you're a first-time contributor ([General GitHub Fork guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo)).
2.  **Clone the repository**.
3.  **Create a new branch** for your contributions.
4.  **Add or update your contributions** using GitHub Desktop, Visual Studio, or VS Code.
5.  **Merge master** back to your branch before pushing.
6.  **Push your changes** to GitHub.
7.  **Submit a Pull Request (PR)** with a detailed description of your changes.  Review the [Pull Request](https://github.com/Azure/Azure-Sentinel/pulls) for comments and make changes as suggested.

### Validation Checks:

*   **Detection Template Structure Validation:** Ensures all required elements are in the YAML files.
*   **KQL Validation:** Validates the syntax of KQL queries within the templates.
*   **Detection Schema Validation:** Validates the schema of a detection.

**For more information on contributions, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).**

### Running Validations Locally:

*   **KQL Validation:** Requires [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download).  Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` and run `dotnet test`.
*   **Detection Schema Validation:**  Requires [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download). Navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` and run `dotnet test`.

### Contributor License Agreement (CLA)

All contributors must agree to a Contributor License Agreement (CLA). You will be prompted by a CLA-bot when submitting a pull request.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).