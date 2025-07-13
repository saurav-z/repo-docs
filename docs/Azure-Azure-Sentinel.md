# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

Secure your digital environment with pre-built detections, exploration queries, hunting queries, workbooks, and playbooks in the [Microsoft Sentinel and Microsoft 365 Defender repository](https://github.com/Azure/Azure-Sentinel), the ultimate resource for cybersecurity professionals.

**Key Features:**

*   **Out-of-the-Box Detections:** Quickly identify potential threats with pre-configured security rules.
*   **Hunting Queries:** Proactively search for malicious activity across your environment. Includes Microsoft 365 Defender hunting queries for advanced threat hunting.
*   **Workbooks:** Visualize your security data and gain insights into your environment's health.
*   **Playbooks:** Automate your security response with pre-built and customizable playbooks.
*   **Comprehensive Coverage:** Enhance your security with content for both Microsoft Sentinel (SIEM) and Microsoft 365 Defender (XDR).

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback & Support

We value your input! Use the following channels to ask questions and provide feedback:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit and vote on features in the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

This project thrives on community contributions. Please review the [Contribution Guidelines](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md) before contributing.

**Contribution Process:**

1.  **Fork the Repository:** If you are a first time contributor, see the [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) before cloning or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
2.  **Create a Branch:** Create a branch for your changes.
3.  **Make Changes:** Add or update content using your preferred method (GitHub website, GitHub Desktop, Visual Studio, VSCode).
4.  **Submit a Pull Request:**
    *   Provide detailed information about your changes in the pull request description.
    *   Check the Pull Request for comments.
    *   Make changes as suggested and update your branch or explain why no change is needed. Resolve the comment when done.

    *   **Detection Template Structure Validation:** All pull requests will undergo validation checks to ensure the structure of the YAML files is correct. See the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how)
    *   **KQL Validation:** The KQL queries in your templates will be validated for syntax. If you are using custom logs table (a table which is not defined on all workspaces by default) you should verify your table schema is defined in json file in the folder *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables*.
    *   **Detection schema validation tests** Similarly to KQL Validation, there is an automatic validation of the schema of a detection.

5.  **CLA:** All contributors must agree to the Contributor License Agreement (CLA).

For more information on what you can contribute and further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.