# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

This repository provides a comprehensive collection of detections, queries, workbooks, and playbooks to help you proactively secure your environment using Microsoft Sentinel and Microsoft 365 Defender.  ([View original repository](https://github.com/Azure/Azure-Sentinel))

**Key Features:**

*   **Pre-built Detections:** Out-of-the-box detections to identify and respond to threats.
*   **Hunting Queries:**  Explore your environment and uncover potential security threats with advanced hunting queries, including Microsoft 365 Defender hunting queries.
*   **Interactive Workbooks:** Visualize your security data with pre-built workbooks for insights and reporting.
*   **Automated Playbooks:** Automate incident response tasks with pre-configured playbooks.
*   **Unified Security:** Leverage resources for both Microsoft Sentinel (SIEM) and Microsoft 365 Defender (XDR).

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

##  Get Involved:  Contribute and Collaborate

We encourage contributions and feedback! Here's how to get involved:

*   **General product-specific Q&A (SIEM & SOAR):** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
*   **General product-specific Q&A (XDR):** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
*   **Feature Requests:** Submit or upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
*   **Report Bugs:**  File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
*   **General Feedback:**  File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project welcomes contributions. To contribute:

1.  **Contributor License Agreement (CLA):** You must agree to a CLA to contribute.  Learn more at [https://cla.microsoft.com](https://cla.microsoft.com).
2.  **Fork and Clone:** Fork the repository and clone it to your local machine.
3.  **Create a Branch:** Create a new branch for your changes.
4.  **Make your changes:** Add or update content.
5.  **Merge Updates:** Merge master into your branch before pushing
6.  **Push Changes:** Push your changes to your forked repository.
7.  **Submit a Pull Request (PR):** Submit a PR with detailed information about your changes.

### PR Validation

*   **Detection Template Structure:** Ensure your detection YAML files adhere to the required structure.
*   **KQL Validation:**  All KQL queries in your PR will be automatically validated for syntax.
*   **Schema Validation:** The schema of the detections will be validated.

#### Running Validations Locally

*   **KQL Validation:** Install the .Net Core 3.1 SDK and run `dotnet test` in the `Azure-Sentinel\\.script\tests\KqlvalidationsTests\` directory.
*   **Detection Schema Validation:** Install the .Net Core 3.1 SDK and run `dotnet test` in the `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\` directory.

Refer to the project's [wiki](https://aka.ms/threathunters) for more contribution details.

### Code of Conduct

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.