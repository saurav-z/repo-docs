# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

This repository provides a wealth of security content, from detections to hunting queries, to help you proactively defend your environment.  Explore pre-built resources and contribute your own expertise to build a stronger security community.  [Visit the original repository.](https://github.com/Azure/Azure-Sentinel)

## Key Features

*   **Pre-built Detections:** Leverage out-of-the-box detections to identify threats in your environment.
*   **Hunting Queries:** Proactively search for threats using advanced hunting queries compatible with both Microsoft Sentinel and Microsoft 365 Defender.
*   **Comprehensive Resources:** Access a library of workbooks, playbooks, and more to streamline your security operations.
*   **Community Driven:** Contribute your own security content and collaborate with a community of security professionals.
*   **Microsoft 365 Defender Integration:**  Benefit from hunting queries that extend to the advanced hunting capabilities within Microsoft 365 Defender.

## Resources

*   [Microsoft Sentinel documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting started with GitHub](https://help.github.com/en#dotcom)

## Get Involved and Contribute

We value your feedback and contributions! Here's how you can participate:

1.  **Ask Questions & Share Ideas:** Engage in [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) and [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
2.  **Suggest Features:**  Provide product-specific feature requests via the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
3.  **Report Issues:**  Report product or contribution bugs using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
4.  **Provide Feedback:** Share general feedback on the community and contribution process with the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).
5.  **Contribute Code:**  Follow the [contribution guidelines](#contribution-guidelines) to submit your detections, queries, and other resources.

## Contribution Guidelines

This project welcomes contributions. Please follow these guidelines:

1.  **Contributor License Agreement (CLA):**  All contributions require a CLA.  Visit [https://cla.microsoft.com](https://cla.microsoft.com) for details.
2.  **Contribution Process:**  Review the [wiki](https://aka.ms/threathunters) for detailed steps.  Contributions typically involve forking the repo, creating a branch, making your changes, and submitting a pull request.
3.  **Pull Request (PR) Requirements:**

    *   Provide clear and concise details about your proposed changes.
    *   Address any comments from reviewers.
    *   PRs are checked for structure validation and KQL validation. Review the output of these checks in the Azure Pipeline to find any errors.

    *   **KQL Validation:** To run KQL validation locally:
        *   Install .Net Core 3.1 SDK
        *   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
        *   Execute `dotnet test`
    *   **Detection Schema Validation:** To run detection schema validation locally:
        *   Install .Net Core 3.1 SDK
        *   Navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
        *   Execute `dotnet test`

4.  **Code of Conduct:** This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.

5.  **Additional Information:** Refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters) for more information.