# Enhance Your Cybersecurity with Microsoft Sentinel and Microsoft 365 Defender

**Secure your environment and proactively hunt for threats with the unified Microsoft Sentinel and Microsoft 365 Defender repository.** This repository is your go-to resource for out-of-the-box detections, exploration queries, hunting queries, workbooks, and playbooks to strengthen your cybersecurity posture.

[Link to Original Repo:](https://github.com/Azure/Azure-Sentinel)

**Key Features:**

*   **Pre-built Content:** Access a wealth of pre-configured detections, queries, workbooks, and playbooks for rapid deployment and threat detection.
*   **Unified Security:** Leverage hunting queries that incorporate both Microsoft Sentinel and Microsoft 365 Defender for comprehensive threat hunting across your environment.
*   **Community Driven:** Benefit from community contributions and resources to stay ahead of evolving threats.
*   **Contribution Welcome:**  Contribute your own detections, queries, and other security content to help others secure their environment.
*   **Extensive Documentation:** Access detailed documentation and guidance for effective utilization and contribution.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

**Get Involved:**

We highly value your input and encourage you to engage with the community:

1.  **Q&A for SIEM and SOAR:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific discussions.
2.  **Q&A for XDR:** Participate in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for expert insights.
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:**  Report product or contribution bugs by filing a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community & Contribution Feedback:** Share general feedback using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

**Contribution Guidelines:**

This project welcomes contributions! Please review the following guidelines to ensure a smooth contribution process:

*   **Contributor License Agreement (CLA):**  All contributions require a CLA, ensuring you have the necessary rights to grant us usage of your contributions. Visit https://cla.microsoft.com for more details.

*   **Contribution Methods:**

    *   **Direct GitHub Upload:** Upload files directly to the repository using the "Upload Files" option within the relevant folder. You will create a branch and submit a Pull Request.
    *   **GitHub Desktop, Visual Studio, or VSCode:**
        1.  Fork the repository.
        2.  Clone the repository.
        3.  Create your own branch.
        4.  Make your changes.
        5.  Merge master into your branch before pushing changes.
        6.  Push your changes.

*   **Pull Request Process:**

    *   After pushing your changes, submit a Pull Request (PR).
    *   Provide detailed explanations of your changes.
    *   Address any comments from reviewers.

    *   **Important PR Checks:**
        *   **Detection Template Structure Validation:** The YAML structure must conform to the required structure, particularly within the "entityMappings" section.  Refer to the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for detailed instructions.
        *   **KQL Validation:**  All KQL queries within the template will undergo syntax validation. Errors may result in a failed test during the PR process. You can view details on the Azure Pipeline.
        *   **Detection Schema Validation** Check for detection's frequency, trigger types and other properties.

    *   **Running KQL Validation Locally:**
        1.  Install .Net Core 3.1 SDK.
        2.  Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
        3.  Run `dotnet test`.

    *   **Running Detection Schema Validation Locally:**
        1.  Install .Net Core 3.1 SDK.
        2.  Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
        3.  Run `dotnet test`.

*   **CLA-bot:** The CLA-bot will guide you through any necessary steps related to the CLA.

*   **Code of Conduct:** This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

*   **Get Started:** For detailed information on contributions and further details, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section of the [wiki](https://aka.ms/threathunters).