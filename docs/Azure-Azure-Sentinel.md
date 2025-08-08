# Microsoft Sentinel and Microsoft 365 Defender Security Content Repository

**Enhance your security posture and proactively hunt for threats with pre-built detections, queries, and more for Microsoft Sentinel and Microsoft 365 Defender.**

[Link to Original Repo: Microsoft Sentinel and Microsoft 365 Defender](https://github.com/Azure/Azure-Sentinel)

This repository serves as a central hub for security professionals to access a wealth of resources designed to accelerate their Microsoft Sentinel and Microsoft 365 Defender deployments. Benefit from out-of-the-box detections, exploration queries, hunting queries (including Microsoft 365 Defender advanced hunting examples), workbooks, and playbooks, all designed to help you secure your environment.

## Key Features:

*   **Pre-built Detections:** Implement instant threat detection across your environment.
*   **Hunting Queries:** Proactively search for threats with ready-to-use queries.
*   **Workbooks & Playbooks:** Streamline your security operations with pre-configured workbooks and automated response playbooks.
*   **Unified Content:** Access resources applicable to both Microsoft Sentinel and Microsoft 365 Defender, including advanced hunting scenarios.
*   **Community Driven:** Contribute to and benefit from a community-driven resource.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support

We value your input. Please use the following channels for questions or feedback:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific Q&A.
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for product-specific Q&A.
3.  **Feature Requests:** Upvote or submit feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project encourages contributions! Before contributing, please familiarize yourself with the [wiki](https://aka.ms/threathunters).

All contributors must agree to the [Contributor License Agreement (CLA)](https://cla.microsoft.com), which grants us the rights to use your contributions.

### How to Contribute:

1.  **Fork the Repository:** If you are a first-time contributor, follow the [GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) before cloning, or follow the [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
2.  **Create a Branch:** Create a new branch for your changes.
3.  **Make Changes:** Add or update content.  Utilize tools like [GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop), [Visual Studio](https://visualstudio.microsoft.com/vs/), or [VSCode](https://code.visualstudio.com/?wt.mc_id=DX_841432).
4.  **Merge and Push:** Merge master back into your branch before pushing your changes.
5.  **Submit a Pull Request (PR):**  Once you push your changes, submit a PR with clear details about your changes. Include a minimal level of detail so a review can clearly understand the reason for the change and what he change is related to in the code.
6.  **Review and Iterate:** Respond to comments and make suggested changes. Resolve comments when you've addressed them.

### Pull Request Validation Checks:

*   **Template Structure Validation:** PRs are checked to ensure that all required parts of the YAML structure are included. See the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for more information.
*   **KQL Validation:** KQL queries are syntax-validated. Errors will be shown in the Azure Pipeline.
    *   If using custom logs, ensure your table schema is defined in a JSON file in the appropriate directory.
    *   Run KQL validation locally by following the instructions detailed in the original README.
*   **Detection Schema Validation:** Automates validation of a detection's schema including frequency and period, trigger type, connector IDs, etc.

### Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

For detailed contribution guidance, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).