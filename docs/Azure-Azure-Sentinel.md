# Microsoft Sentinel and Microsoft 365 Defender: Your One-Stop Shop for Security Content

**Enhance your security posture and proactively hunt for threats with this comprehensive repository of detections, queries, workbooks, and playbooks for Microsoft Sentinel and Microsoft 365 Defender.**  This repository, maintained by Microsoft, provides a wealth of resources to help you quickly get up and running with these powerful security tools and continuously improve your threat detection and response capabilities.

[Link to Original Repository: https://github.com/Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel)

**Key Features:**

*   **Pre-built Detections:** Implement out-of-the-box detections to identify and respond to threats quickly.
*   **Hunting Queries:** Proactively search for malicious activity in your environment using advanced hunting queries for both Microsoft Sentinel and Microsoft 365 Defender.
*   **Exploration Queries:** Gain valuable insights into your data with pre-built queries to explore and understand your security landscape.
*   **Workbooks:** Visualize your security data and gain insights using interactive workbooks.
*   **Playbooks:** Automate security tasks and streamline incident response with ready-to-use playbooks.
*   **Microsoft 365 Defender Integration:** Leverage advanced hunting scenarios and data integration for a unified security approach.
*   **Community-Driven Content:** Benefit from contributions from the community, including new samples and resources.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support

We value your input and are committed to providing excellent support.  Here's how to get help and provide feedback:

1.  **SIEM and SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:**  Submit and vote on feature requests in the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:**  File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contribution Guidelines

This project welcomes contributions! Before contributing, please review the following guidelines:

1.  **Contributor License Agreement (CLA):** You must agree to a CLA, granting Microsoft the rights to use your contributions.  Visit [https://cla.microsoft.com](https://cla.microsoft.com) for details.
2.  **Fork and Clone the Repository:**  Fork the repository and clone it locally to make your changes.  [General GitHub Fork the repo guidance](https://docs.github.com/github/getting-started-with-github/fork-a-repo) or [Specific steps for the Sentinel repo](https://github.com/Azure/Azure-Sentinel/blob/master/GettingStarted.md).
3.  **Create a Branch:** Create a new branch for your work.
4.  **Make Your Changes:**  Add or update content.
5.  **Merge and Push:**  Merge the master branch back into your branch before pushing.
6.  **Submit a Pull Request:** Submit a Pull Request (PR) with detailed information about your changes.
7.  **Review and Address Comments:**  Check the PR for comments and make any necessary changes.

### Pull Request Validation
As part of the PR checks we run a structure validation to make sure all required parts of the YAML structure are included.  For Detections, there is a new section that must be included.  See the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for more information.  If this section or any other required section is not included, then a validation error will occur.

### Pull Request KQL Validation Check
As part of the PR checks we run a syntax validation of the KQL queries defined in the template. If this check fails go to Azure Pipeline (by pressing on the errors link on the checks tab in your PR).
![Azurepipeline](.github/Media/Azurepipeline.png)
In the pipeline you can see which test failed and what is the cause:
![Pipeline Tests Tab](.github/Media/PipelineTestsTab.png)

### Run KQL Validation Locally
In order to run the KQL validation before submitting Pull Request in you local machine:
* You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
* Execute `dotnet test`

### Detection schema validation tests
Similarly to KQL Validation, there is an automatic validation of the schema of a detection.
The schema validation includes the detection's frequency and period, the detection's trigger type and threshold, validity of connectors Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)), etc.
A wrong format or missing attributes will result with an informative check failure, which should guide you through the resolution of the issue, but make sure to look into the format of already approved detection.

### Run Detection Schema Validation Locally
In order to run the KQL validation before submitting Pull Request in you local machine:
* You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
* Execute `dotnet test`


## Code of Conduct

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

## Get Started

For more details on contributing and the types of content you can provide, see the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).