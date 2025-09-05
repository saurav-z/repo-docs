# Microsoft Sentinel & Microsoft 365 Defender: Your Unified Security Powerhouse

**Enhance your security posture with a vast library of detections, queries, workbooks, and playbooks for Microsoft Sentinel and Microsoft 365 Defender, empowering you to proactively hunt for threats.** Visit the [original repository](https://github.com/Azure/Azure-Sentinel) to get started.

## Key Features:

*   **Pre-built Detections:** Leverage out-of-the-box detections to identify and respond to security threats.
*   **Hunting Queries:** Explore and hunt for threats across your environment with specialized queries, including Microsoft 365 Defender advanced hunting scenarios.
*   **Comprehensive Resources:** Access workbooks, playbooks, and other resources to accelerate your Microsoft Sentinel and Microsoft 365 Defender deployments.
*   **Community-Driven:** Benefit from contributions from security experts and a community dedicated to enhancing your security capabilities.
*   **Unified Security:** Integrate with Microsoft 365 Defender for a cohesive and integrated security approach.

## Resources:

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support:

We value your input. Connect with the community and share your questions and feedback through these channels:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and vote on feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Bug Reports:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community & Contribution Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines:

This project encourages contributions. Please review the following guidelines to ensure a smooth contribution process:

*   **Contributor License Agreement (CLA):** Contributions require you to agree to a Contributor License Agreement (CLA).  Learn more at [https://cla.microsoft.com](https://cla.microsoft.com).
*   **Getting Started:** If you are new, please follow the [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) before cloning or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).

### Contributing to GitHub:

1.  **Submit Directly (for small updates):**
    *   Browse to the folder.
    *   Use "Upload Files."
    *   Create a branch and submit a Pull Request.
2.  **Use GitHub Desktop/VSCode/VS:**
    *   Fork the repository.
    *   Clone the repository.
    *   Create a branch.
    *   Make your changes.
    *   Merge master back to your branch.
    *   Push your changes.
3.  **Pull Request (PR):**
    *   Submit a PR with details about the changes.
    *   Address any comments.

### PR Validation:

*   **Template Structure Validation:**  Ensure the correct YAML structure.  See the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for details.
*   **KQL Validation:**  KQL queries will be validated.  If failures, check the Azure Pipeline.
    *   If using custom logs, define your table schema in `Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables`.
    *   Run KQL validation locally using: `dotnet test` from `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
*   **Detection Schema Validation:** The schema of a detection will be validated, including frequency, trigger type, etc.
    *   Run detection schema validation locally using: `dotnet test` from  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`

When submitting a pull request, the CLA-bot will guide you regarding the CLA.

### Code of Conduct:

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For contribution details, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the [wiki](https://aka.ms/threathunters).
```
Key improvements and optimizations:

*   **Stronger Hook:**  The one-sentence hook highlights the main benefit of the repository.
*   **SEO-Friendly Headings:** Uses clear, descriptive headings (e.g., "Key Features," "Resources") for better organization and searchability.
*   **Bulleted Key Features:**  Uses bullets to highlight key benefits and attract readers.
*   **Concise Language:** Streamlines the text for better readability.
*   **Clear Calls to Action:** Encourages users to explore the repository and contribute.
*   **Links:**  Ensures all relevant links are included.
*   **Keywords:** The summary uses relevant keywords like "Microsoft Sentinel," "Microsoft 365 Defender," "detections," "queries," "threat hunting," "security," and "SOAR."
*   **Improved Formatting:** Uses Markdown for better readability.
*   **Comprehensive:** Includes all critical information from the original README, including contribution guidelines, feedback channels, and resource links.
*   **Contribution instructions are expanded** to offer clarity.
*   **Local validation instructions are added**.