# Enhance Your Security Posture with Microsoft Sentinel & Microsoft 365 Defender

**This repository is your go-to resource for ready-to-use security content, empowering you to detect threats, hunt for adversaries, and fortify your environment with Microsoft Sentinel and Microsoft 365 Defender.**  [Explore the original repository](https://github.com/Azure/Azure-Sentinel)

This repository provides a wealth of resources to help you get started with Microsoft Sentinel and Microsoft 365 Defender. It offers a unified collection of detections, exploration queries, hunting queries, workbooks, and playbooks, all designed to enhance your security posture. This includes both Microsoft Sentinel and Microsoft 365 Defender hunting queries for advanced threat hunting capabilities.

**Key Features:**

*   **Out-of-the-Box Detections:** Immediately identify potential threats with pre-built detection rules.
*   **Hunting Queries:** Proactively search for malicious activity with custom queries.
*   **Exploration Queries:** Deep dive into your data to uncover hidden insights.
*   **Workbooks:** Visualize your security data for improved understanding and reporting.
*   **Playbooks:** Automate your security responses for faster incident handling.
*   **Unified Security Content:** Leverages content for both Microsoft Sentinel and Microsoft 365 Defender.

**Resources:**

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

**Feedback and Support:**

We value your input! Here's how to connect with the community:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for general product Q&A.
2.  **XDR Q&A:** Participate in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for general XDR product Q&A.
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** Use the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=) to provide feedback on the community and contribution process.

**Contribution Guidelines**

This project welcomes contributions! Please review the following guidelines before submitting:

*   **Contributor License Agreement (CLA):**  Ensure you agree to the CLA; details are at https://cla.microsoft.com.
*   **GitHub Forking:** If you're new to the repository, please reference the [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md) before cloning or forking.

**Contribution Process:**

1.  **Fork & Clone:** Fork the repository and clone it to your local machine.
2.  **Create a Branch:** Create a new branch for your changes.
3.  **Make Changes:** Add or update content in your branch.
4.  **Merge:** Merge the master branch back into your branch before pushing your changes.
5.  **Pull Request:** Submit a Pull Request (PR) with details about your changes. Provide sufficient detail for review.
6.  **Review & Iterate:**  Address comments on your PR and make necessary changes.
7.  **Validation:**
    *   **Detection Template Structure Validation:** Required YAML structure for detection templates.
    *   **KQL Validation:** Checks KQL queries for syntax using `dotnet test` command or Azure Pipeline
    *   **Detection Schema Validation:** Automatic validation of the detection schema and format.

**Code of Conduct:**

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.

**Additional Resources:**

*   Refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters) for more information.