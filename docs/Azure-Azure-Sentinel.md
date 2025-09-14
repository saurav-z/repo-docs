# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

**Secure your environment and proactively hunt for threats with the comprehensive resources available in this repository, providing a unified approach to threat detection and response.** This repository houses a wealth of security content, including detections, queries, workbooks, and playbooks, designed to help you maximize the value of Microsoft Sentinel and Microsoft 365 Defender.

[Link to Original Repo:  https://github.com/Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel)

## Key Features:

*   **Out-of-the-Box Detections:** Immediately identify potential threats with pre-built detection rules.
*   **Exploration & Hunting Queries:**  Uncover malicious activity with powerful KQL queries for proactive threat hunting.
*   **Microsoft 365 Defender Integration:**  Leverage advanced hunting capabilities across Microsoft 365 Defender and Sentinel.
*   **Workbooks & Playbooks:** Visualize data and automate incident response with pre-built resources.
*   **Community-Driven:** Contribute and access community-created content to enhance your security.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Get Involved & Provide Feedback

Your feedback is valuable!  Here's how to connect with the community and the product teams:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel)
2.  **XDR Q&A:**  Engage in the [Microsoft 365 Defender Tech Community](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection)
3.  **Feature Requests:** Submit and upvote on the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8)
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=)
5.  **General Feedback:**  File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=)

## Contributing Guidelines

This project welcomes contributions! Please review these guidelines:

1.  **Contributor License Agreement (CLA):**  All contributions require agreement to the CLA.  Visit [https://cla.microsoft.com](https://cla.microsoft.com) for details.

2.  **Contribution Process:**

    *   **General Steps:**
        *   Fork the repository.
        *   Clone the repository.
        *   Create a new branch for your work.
        *   Make your changes.
        *   Merge master back to your branch.
        *   Push your changes to GitHub.
        *   Submit a Pull Request (PR).
    *   **Adding Contributions via the GitHub Website**
        *   Browse to the folder you want to upload your file to
        *   Choose Upload Files and browse to your file.
        *   You will be required to create your own branch and then submit the Pull Request for review.
    *   **PR Requirements:** Include details on the proposed changes.  Clearly explain the reason for the change.
    *   **PR Checks:** Ensure your submission passes the structure validation and KQL validation checks performed as part of the PR process.  See the detailed instructions for troubleshooting validation failures.

3.  **KQL Validation:**  Learn how to validate KQL queries locally before submitting your pull request.

    *   Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
    *   Navigate to `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
    *   Run `dotnet test`
4.  **Detection Schema Validation:** Learn how to validate detection schema locally before submitting your pull request.

    *   Install [.Net Core 3.1 SDK](https://dotnet.microsoft.com/download)
    *   Navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
    *   Run `dotnet test`

Refer to the [wiki](https://aka.ms/threathunters) for more detailed information.

### Pull Request Guidelines
* **Pull Request Detection Template Structure Validation Check**: As part of the PR checks we run a structure validation to make sure all required parts of the YAML structure are included.  For Detections, there is a new section that must be included.  See the [contribution guidelines](https://github.com/Azure/Azure-Sentinel/wiki/Contribute-to-Sentinel-GitHub-Community-of-Queries#now-onto-the-how) for more information.  If this section or any other required section is not included, then a validation error will occur similar to the below.
The example is specifically if the YAML is missing the entityMappings section:

```
A total of 1 test files matched the specified pattern.
[xUnit.net 00:00:00.95]     Kqlvalidations.Tests.DetectionTemplateStructureValidationTests.Validate_DetectionTemplates_HaveValidTemplateStructure(detectionsYamlFileName: "ExcessiveBlockedTrafficGeneratedbyUser.yaml") [FAIL]
  X Kqlvalidations.Tests.DetectionTemplateStructureValidationTests.Validate_DetectionTemplates_HaveValidTemplateStructure(detectionsYamlFileName: "ExcessiveBlockedTrafficGeneratedbyUser.yaml") [104ms]
  Error Message:
   Expected object to be <null>, but found System.ComponentModel.DataAnnotations.ValidationException with message "An old mapping for entity 'AccountCustomEntity' does not have a matching new mapping entry."
```

### Pull Request KQL Validation Check
As part of the PR checks we run a syntax validation of the KQL queries defined in the template. If this check fails go to Azure Pipeline (by pressing on the errors link on the checks tab in your PR)
![Azurepipeline](.github/Media/Azurepipeline.png)
In the pipeline you can see which test failed and what is the cause:
![Pipeline Tests Tab](.github/Media/PipelineTestsTab.png)

Example error message:
```
A total of 1 test files matched the specified pattern.
[xUnit.net 00:00:01.81]     Kqlvalidations.Tests.KqlValidationTests.Validate_DetectionQueries_HaveValidKql(detectionsYamlFileName: "ExcessiveBlockedTrafficGeneratedbyUser.yaml") [FAIL]
  X Kqlvalidations.Tests.KqlValidationTests.Validate_DetectionQueries_HaveValidKql(detectionsYamlFileName: "ExcessiveBlockedTrafficGeneratedbyUser.yaml") [21ms]
  Error Message:
   Template Id:fa0ab69c-7124-4f62-acdd-61017cf6ce89 is not valid Errors:The name 'SymantecEndpointProtection' does not refer to any known table, tabular variable or function., Code: 'KS204', Severity: 'Error', Location: '67..93',The name 'SymantecEndpointProtection' does not refer to any known table, tabular variable or function., Code: 'KS204', Severity: 'Error', Location: '289..315'
```
If you are using custom logs table (a table which is not defined on all workspaces by default) you should verify
your table schema is defined in json file in the folder *Azure-Sentinel\\.script\tests\KqlvalidationsTests\CustomTables*

**Example for table tablexyz.json**
```json
{
  "Name": "tablexyz",
  "Properties": [
    {
      "Name": "SomeDateTimeColumn",
      "Type": "DateTime"
    },
    {
      "Name": "SomeStringColumn",
      "Type": "String"
    },
    {
      "Name": "SomeDynamicColumn",
      "Type": "Dynamic"
    }
  ]
}
```
### Run KQL Validation Locally
In order to run the KQL validation before submitting Pull Request in you local machine:
* You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\KqlvalidationsTests\`
* Execute `dotnet test`

Example of output (in Ubuntu):
```
Welcome to .NET Core 3.1!
----------------------
SDK Version: 3.1.403

Telemetry
---------
The .NET Core tools collect usage data in order to help us improve your experience. The data is anonymous. It is collected by Microsoft and shared with the community. You can opt-out of telemetry by setting the DOTNET_CLI_TELEMETRY_OPTOUT environment variable to '1' or 'true' using your favorite shell.

Read more about .NET Core CLI Tools telemetry: https://aka.ms/dotnet-cli-telemetry

----------------
Explore documentation: https://aka.ms/dotnet-docs
Report issues and find source on GitHub: https://github.com/dotnet/core
Find out what's new: https://aka.ms/dotnet-whats-new
Learn about the installed HTTPS developer cert: https://aka.ms/aspnet-core-https
Use 'dotnet --help' to see available commands or visit: https://aka.ms/dotnet-cli-docs
Write your first app: https://aka.ms/first-net-core-app
--------------------------------------------------------------------------------------
Test run for /mnt/c/git/Azure-Sentinel/.script/tests/KqlvalidationsTests/bin/Debug/netcoreapp3.1/Kqlvalidations.Tests.dll(.NETCoreApp,Version=v3.1)
Microsoft (R) Test Execution Command Line Tool Version 16.7.0
Copyright (c) Microsoft Corporation.  All rights reserved.

Starting test execution, please wait...

A total of 1 test files matched the specified pattern.

Test Run Successful.
Total tests: 171
     Passed: 171
 Total time: 25.7973 Seconds
```

### Detection schema validation tests
Similarly to KQL Validation, there is an automatic validation of the schema of a detection.
The schema validation includes the detection's frequency and period, the detection's trigger type and threshold, validity of connectors Ids ([valid connectors Ids list](https://github.com/Azure/Azure-Sentinel/blob/master/.script/tests/detectionTemplateSchemaValidation/ValidConnectorIds.json)), etc.
A wrong format or missing attributes will result with an informative check failure, which should guide you through the resolution of the issue, but make sure to look into the format of already approved detection.

### Run Detection Schema Validation Locally
In order to run the KQL validation before submitting Pull Request in you local machine:
* You need to have **.Net Core 3.1 SDK** installed [How to download .Net](https://dotnet.microsoft.com/download) (Supports all platforms)
* Open Shell and navigate to  `Azure-Sentinel\\.script\tests\DetectionTemplateSchemaValidation\`
* Execute `dotnet test`

4.  **CLA-bot:**  The CLA-bot will automatically determine if you need to provide a CLA.  Follow its instructions.

5.  **Code of Conduct:**  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

For more details, explore the ["get started"](https://github.com/Azure/Azure-Sentinel/wiki#get-started) section on the project's [wiki](https://aka.ms/threathunters).