# Microsoft Sentinel and Microsoft 365 Defender: Enhance Your Security Posture

This repository offers a wealth of pre-built security content, including detections, queries, workbooks, and playbooks, to help you proactively secure your environment with Microsoft Sentinel and Microsoft 365 Defender.  [Visit the original repository](https://github.com/Azure/Azure-Sentinel) for the latest updates and contributions.

## Key Features

*   **Out-of-the-Box Detections:** Implement pre-configured security alerts and threat detection rules.
*   **Hunting Queries:**  Proactively search for threats using pre-built hunting queries, including Microsoft 365 Defender advanced hunting scenarios.
*   **Workbooks and Playbooks:**  Visualize data and automate responses to security incidents.
*   **Unified Security Content:** Access resources for both Microsoft Sentinel (SIEM) and Microsoft 365 Defender (XDR) to streamline your security operations.
*   **Community Driven:** Benefit from a community-driven resource, updated regularly with new detections and content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Feedback & Community

We value your input!  Here's how to connect with the community:

1.  **SIEM & SOAR Q&A:**  Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific discussions.
2.  **XDR Q&A:**  Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR related questions and answers.
3.  **Feature Requests:**  Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:**  Report product or contribution bugs via a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community Feedback:**  Provide general feedback on community and contribution process via a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contributing

We welcome contributions! Please review the following guidelines to help improve the quality of the contributions.

1.  **Contributor License Agreement (CLA):**  All contributions require agreeing to a CLA.  Learn more at https://cla.microsoft.com.
2.  **Contribution Methods:**
    *   **Direct GitHub Upload:** Upload files directly through the GitHub interface within the specific folder.
    *   **GitHub Desktop/VS Code/Visual Studio:** Use a local development environment and follow these steps:
        *   Fork the repository.
        *   Clone the repository.
        *   Create a new branch for your changes.
        *   Make additions/updates.
        *   Merge the master branch into your branch before pushing.
        *   Push your changes.
        *   Submit a Pull Request.
3.  **Pull Request (PR) Guidelines:**
    *   Include detailed descriptions of the proposed changes.
    *   Address comments and make necessary changes.
4.  **Validation Checks:**
    *   **Detection Template Structure Validation:**  Ensure all required sections of the YAML structure are included, especially the new entityMappings section.
    *   **KQL Validation:**  Validate the syntax of your KQL queries. Fix errors such as table name mismatches. If using custom logs, ensure your table schema is defined in the directory structure of the project.
    *   **Detection Schema Validation:** Verify the format of the detection definition including frequency, period, and connector ids and resolve any failures.
5.  **Local Validation:**
    *   Use .Net Core 3.1 SDK to execute the validation and follow the test output.
6.  **CLA-bot:** A CLA-bot will guide you through the CLA process when you submit a pull request.

## Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For detailed contribution instructions, see the [wiki](https://aka.ms/threathunters).