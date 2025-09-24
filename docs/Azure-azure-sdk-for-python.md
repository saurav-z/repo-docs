# Azure SDK for Python: Simplify Cloud Development

**Get started with the Azure SDK for Python and unlock powerful cloud capabilities.** [(Original Repository)](https://github.com/Azure/azure-sdk-for-python)

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository is for the active development of the Azure SDK for Python.  It provides comprehensive libraries for interacting with a wide range of Azure services, enabling developers to build robust and scalable cloud applications.

## Key Features:

*   **Comprehensive Service Coverage:** Access a vast array of Azure services, from compute and storage to AI and databases.
*   **Client Libraries:** Utilize separate, focused libraries for each service for improved modularity and ease of use.
*   **Modern Design:** Leverage libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for a consistent and intuitive development experience.
*   **New and Previous Versions:** Access both new and previous versions of packages, including GA and preview releases, to meet your specific needs and production requirements.
*   **Management Libraries:** Provision and manage Azure resources with dedicated management libraries.
*   **Cross-Platform Compatibility:** Build and run applications on Python 3.9 or later.
*   **Authentication and Authorization**: Seamlessly integrated authentication using Azure Identity.
*   **Telemetry Configuration**: Options to opt out or configure the telemetry settings

## Getting Started

Each service has a separate set of libraries.  Refer to the `README.md` (or `README.rst`) file within each library's project folder in the `/sdk` directory for instructions.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages

Explore the available libraries, categorized for your convenience:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

Benefit from the latest packages, including GA and preview releases, which offer the newest features and improvements. These libraries are designed for consuming existing Azure resources. They share core functionalities like retries, logging, and authentication. Find the [most up-to-date package list here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:**  For production use, choose stable, non-preview libraries.

### Client: Previous Versions

Utilize the last stable versions of packages. These offer production-ready functionality and provide wide service coverage.

### Management: New Releases

Use the latest management libraries, adhering to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), which offers:
*   Azure Identity library
*   HTTP pipeline with custom policies
*   Error handling
*   Distributed tracing

Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt). A migration guide is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:**  Use stable, non-preview libraries for production.  If you encounter authentication issues after upgrading, refer to the migration guide.

### Management: Previous Versions

Find a complete list of management libraries for provisioning and managing Azure resources [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

Management libraries are identifiable by `azure-mgmt-` namespaces (e.g., `azure-mgmt-compute`).

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Stack Overflow using the `azure` and `python` tags.

## Data Collection & Telemetry

The SDK collects information about your use of the software and sends it to Microsoft for service improvement. Review the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default. To opt-out, see the example in the original README.

## Reporting Security Issues

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.  All contributions require a Contributor License Agreement (CLA).

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).