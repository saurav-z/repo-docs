# Enhance Your Security Posture with Microsoft Sentinel and Microsoft 365 Defender

**This repository provides a comprehensive collection of security content, including detections, queries, workbooks, and playbooks, to help you proactively defend your environment with Microsoft Sentinel and Microsoft 365 Defender.** Access the original repo [here](https://github.com/Azure/Azure-Sentinel).

## Key Features

*   **Out-of-the-Box Detections:** Implement pre-built rules to identify and respond to threats immediately.
*   **Hunting Queries:** Leverage advanced queries for proactive threat hunting across Microsoft 365 Defender and Sentinel.
*   **Workbooks:** Utilize pre-built workbooks for data visualization and insightful security analysis.
*   **Playbooks:** Automate incident response and security tasks with ready-to-use playbooks.
*   **Unified Security:** Combine Microsoft Sentinel (SIEM) and Microsoft 365 Defender (XDR) for a comprehensive security approach.
*   **Community Driven:** Leverage content from the community and contribute your own detections and queries.

## Resources

*   [Microsoft Sentinel Documentation](https://go.microsoft.com/fwlink/?linkid=2073774&clcid=0x409)
*   [Microsoft 365 Defender Documentation](https://docs.microsoft.com/microsoft-365/security/defender/microsoft-365-defender?view=o365-worldwide)
*   [Security Community Webinars](https://aka.ms/securitywebinars)
*   [Getting Started with GitHub](https://help.github.com/en#dotcom)

## Feedback and Support

We value your feedback. Here are ways to get help:

1.  **SIEM & SOAR Q&A:** Join the [Microsoft Sentinel Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-sentinel/bd-p/MicrosoftSentinel).
2.  **XDR Q&A:** Join the [Microsoft 365 Defender Tech Community conversations](https://techcommunity.microsoft.com/t5/microsoft-365-defender/bd-p/MicrosoftThreatProtection).
3.  **Feature Requests:** Submit and vote on feature requests in the [Microsoft Sentinel feedback forums](https://feedback.azure.com/d365community/forum/37638d17-0625-ec11-b6e6-000d3a4f07b8).
4.  **Report Bugs:** File a GitHub Issue using the [Bug template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=bug_report.md&title=).
5.  **General Feedback:** File a GitHub Issue using the [Feature Request template](https://github.com/Azure/Azure-Sentinel/issues/new?assignees=&labels=&template=feature_request.md&title=).

## Contributing Guidelines

This project welcomes contributions. To contribute:

1.  **Review the [wiki](https://aka.ms/threathunters)** for detailed contribution instructions.
2.  **Fork the repository** and create a branch for your contributions.
3.  **Submit Pull Requests** with your changes, following the structure guidelines.
4.  **Ensure KQL validation** passes before submitting your pull request. If it does not, see the [KQL Validation Check](#pull-request-kql-validation-check) section below for details on fixing it.
5.  **Ensure Detection Schema Validation** passes before submitting your pull request. If it does not, see the [Detection Schema Validation Tests](#detection-schema-validation-tests) for details on fixing it.
6.  Adhere to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

### Pull Request KQL Validation Check
As part of the PR checks we run a syntax validation of the KQL queries defined in the template. If this check fails go to Azure Pipeline (by pressing on the errors link on the checks tab in your PR).
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

### CLA Requirement

All contributions require a Contributor License Agreement (CLA). Follow the instructions provided by the CLA-bot when submitting a pull request.

For any other queries, you can contact [AzureSentinel@microsoft.com](mailto:AzureSentinel@microsoft.com).