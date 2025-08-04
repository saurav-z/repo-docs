# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

This repository is your go-to resource for detections, queries, workbooks, and playbooks to elevate your security posture with Microsoft Sentinel and Microsoft 365 Defender. [Explore the full repository here](https://github.com/Azure/Azure-Sentinel).

## Key Features:

*   **Out-of-the-Box Detections:** Quickly identify and respond to potential threats.
*   **Exploration and Hunting Queries:** Empower your security team with advanced threat hunting capabilities.
*   **Comprehensive Workbooks:** Visualize your security data and gain actionable insights.
*   **Automated Playbooks:** Streamline your incident response with automated workflows.
*   **Microsoft 365 Defender Integration:** Leverage advanced hunting queries for a unified security approach.
*   **Community-Driven Content:** Access and contribute to a growing library of security content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback

We value your contributions and feedback! Join the community to ask questions, provide feedback, and request new content.

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Upvote or post new requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

We welcome contributions! Please review the following guidelines before submitting your work.

### Contribution Requirements

*   **Contributor License Agreement (CLA):** You must agree to a CLA. Details are available at [https://cla.microsoft.com](https://cla.microsoft.com).

### Steps to Contribute:

1.  **Fork the Repository:**  If you are a first time contributor, [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) before cloning or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
2.  **Clone the Repository:** Clone the repository to your local machine.
3.  **Create a Branch:** Create a new branch for your changes.
4.  **Make Your Changes:**  Add or update content using [GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop), [Visual Studio](https://visualstudio.microsoft.com/vs/), or [VSCode](https://code.visualstudio.com/?wt.mc_id=DX_841432).
5.  **Merge Master:** Before you push your changes make sure to merge master back to your branch.
6.  **Push Changes:** Push your branch to your forked repository on GitHub.
7.  **Submit a Pull Request:** Submit a Pull Request (PR) with details about your proposed changes.

### Pull Request Checks

*   **Detection Template Structure Validation:** The PR will run structure validation checks to ensure your contributions adhere to the required YAML structure.  See the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for more information.
*   **KQL Validation:** KQL queries within your PR will undergo syntax validation.
*   **Detection Schema Validation:** The schema of your detection will undergo automatic validation to ensure correct formatting and attribute usage.

### Run Validations Locally:

*   **KQL Validation:**
    *   Install the [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download).
    *   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Run `dotnet test`.
*   **Detection Schema Validation:**
    *   Install the [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download).
    *   Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Run `dotnet test`.

### Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  For questions, contact [opencode@microsoft.com](mailto:opencode@microsoft.com).

For more details on contributing and content, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).