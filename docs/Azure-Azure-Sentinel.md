# **Azure Sentinel & Microsoft 365 Defender: Your Unified Security Hub**

**Secure your environment and proactively hunt for threats with the collaborative power of Azure Sentinel and Microsoft 365 Defender!** This repository provides a wealth of pre-built security content, including detections, queries, workbooks, and playbooks, to help you get started quickly. Access a treasure trove of resources and contribute to a community-driven approach to cybersecurity.

*   **Out-of-the-Box Detections:** Immediately identify potential threats with pre-configured detection rules.
*   **Hunting Queries:** Proactively search for threats using advanced hunting queries, including Microsoft 365 Defender integrations.
*   **Customizable Workbooks:** Gain insights through interactive dashboards and data visualizations.
*   **Automated Playbooks:** Streamline your security operations with automated response actions.
*   **Community Driven:** Leverage the power of community contributions and collaborate on new security content.

## **Key Resources**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
*   [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
*   [Microsoft Sentinel Feedback Forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)

## **Contribution Guidelines**

This project thrives on contributions! Whether you're a seasoned security professional or just starting out, your insights are welcome. Please review the [wiki](https://aka.ms/threathunters) for guidance on getting started.

### **How to Contribute**

1.  **Fork and Clone:** Fork the repository and clone it to your local machine.  Follow the [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) before cloning or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
2.  **Create a Branch:** Create a new branch for your changes.
3.  **Make Edits:** Add or update detections, queries, playbooks, or any other valuable security content.
4.  **Submit a Pull Request:** Submit a pull request to propose your changes.  Be sure to include clear details about your changes.

### **Important Checks for Pull Requests**

*   **Detection Schema Validation:** Ensure your detection templates adhere to the required structure.  See the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for more information.
*   **KQL Validation:** Validate your KQL queries to ensure they are syntactically correct.  If using custom logs, verify your table schema is defined in the appropriate JSON file (see the original documentation for details.)
*   **Detection Schema Validation Locally:** In order to run the schema validation before submitting Pull Request in you local machine:
    * You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
    * Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    * Execute `dotnet test`

### **CLA Requirement**

All contributions require you to agree to a Contributor License Agreement (CLA). You will be guided through this process by the CLA-bot when you submit your pull request.

### **Code of Conduct**

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

**Ready to get started?** Explore the repository, contribute your expertise, and help build a stronger security community.  For further information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).

[**Back to the main repository**](https://github.com/Azure/Azure-Sentinel)