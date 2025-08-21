# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

**Secure your environment and proactively hunt for threats with a unified repository of detections, queries, and more for Microsoft Sentinel and Microsoft 365 Defender.** ([Original Repository](https://github.com/Azure/Azure-Sentinel))

This repository is your one-stop shop for security content, providing valuable resources to get you up and running with Microsoft Sentinel and Microsoft 365 Defender, including:

*   **Out-of-the-Box Detections:** Ready-to-deploy rules to identify and respond to threats.
*   **Exploration Queries:** Powerful queries to investigate security incidents and gain deeper insights.
*   **Hunting Queries:** Advanced queries for proactive threat hunting across your environment.
*   **Workbooks:** Pre-built dashboards for visualizing security data and monitoring key metrics.
*   **Playbooks:** Automated workflows to streamline incident response and remediation.
*   **Microsoft 365 Defender Integration:** Hunting queries that extend your threat detection capabilities to Microsoft 365 Defender.

## Key Features

*   **Unified Security Content:** Access a comprehensive library of security content for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Proactive Threat Hunting:** Leverage hunting queries to discover and neutralize threats before they cause damage.
*   **Community-Driven:** Benefit from contributions from the security community and submit your own to enhance the repository.
*   **Easy Onboarding:** Get up and running quickly with pre-built detections, queries, and workbooks.
*   **Continuous Updates:** Stay ahead of emerging threats with regularly updated content.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support

We value your feedback and are here to help. Connect with the community and provide input through the following channels:

1.  **SIEM/SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel) for product-specific questions.
2.  **XDR Q&A:** Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection) for XDR-related inquiries.
3.  **Feature Requests:** Submit and upvote feature requests on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Bug Reports:** Report product or contribution bugs by filing a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **Community Feedback:** Provide general feedback on the community and contribution process via the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contribution Guidelines

This project welcomes contributions and suggestions to help improve its capabilities and ensure the best content is available.

### How to Contribute

1.  **Fork the Repository:** Create your own fork of the repository.
2.  **Create a Branch:** Create a branch in your fork for your changes.
3.  **Make Changes:** Add your new or updated content.  Refer to the [wiki](https://aka.ms/threathunters) for further details.
4.  **Submit a Pull Request:** Create a pull request with details on your changes.
5.  **Pass Validation:** The pull request process will validate the KQL and template structures for correctness. See below for examples of how to prepare for this process.
6.  **Code of Conduct:** All contributors agree to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

### Pull Request Validation Details
As part of the PR checks we run structure and syntax validation to ensure all required parts of the structure and syntax are included in the code. 
**Detection Template Structure Validation Check:**

*   Checks to make sure all required parts of the YAML structure are included.  
*   See the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for more information.  
**Pull Request KQL Validation Check:**

*   Syntax validation of the KQL queries defined in the template.
*   Errors in the syntax must be addressed before you can submit a valid PR.

### Run KQL Validation Locally
In order to run the KQL validation before submitting Pull Request in your local machine:
*   You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
*   Open Shell and navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
*   Execute `dotnet test`

### Detection schema validation tests
*   Similarly to KQL Validation, there is an automatic validation of the schema of a detection.
*   The schema validation includes the detection's frequency and period, the detection's trigger type and threshold, validity of connectors Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)), etc.

### Run Detection Schema Validation Locally
In order to run the KQL validation before submitting Pull Request in you local machine:
*   You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
*   Open Shell and navigate to `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
*   Execute `dotnet test`

For more information, refer to the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).