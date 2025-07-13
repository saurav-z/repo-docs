# Azure SDK for Python: Build Powerful Python Applications for Azure

**Develop robust and scalable Python applications that seamlessly integrate with Microsoft Azure using the official Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository is the hub for the active development of the Azure SDK for Python. Access comprehensive [developer documentation](https://docs.microsoft.com/python/azure/) and [versioned documentation](https://azure.github.io/azure-sdk-for-python) for optimal usage.

**Key Features:**

*   **Comprehensive Library Support:** Access individual libraries tailored for specific Azure services, simplifying your project dependencies.
*   **Latest Releases & Previews:** Explore the newest GA and preview releases, built to consume and interact with Azure resources.
*   **Management Libraries:** Provision and manage Azure resources effectively with management libraries.
*   **Production-Ready Stability:** Utilize stable, non-preview libraries for reliable production deployments.
*   **Azure SDK Design Guidelines:** Benefit from libraries adhering to the Azure SDK Design Guidelines for Python, including Azure Identity and HTTP Pipelines.
*   **Detailed Documentation and Examples:** Access detailed documentation and code samples to get started quickly.

## Getting Started

To begin, choose from the individual service libraries located in the `/sdk` directory. Each library has a `README.md` (or `README.rst`) file for specific instructions.

### Prerequisites

The client libraries support Python 3.9 or later. Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Packages Available

Libraries are categorized as:
*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest packages, including GA and preview releases.  They provide functionalities like retries, logging, transport, and authentication.  Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> NOTE: For production, use stable, non-preview libraries.

### Client: Previous Versions

These are the last stable versions of production-ready packages.

### Management: New Releases

A set of management libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> NOTE:  Use stable libraries for production.  Refer to the migration guide if you're upgrading.

### Management: Previous Versions

Find a list of these libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries start with `azure-mgmt-`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **StackOverflow:** Search StackOverflow using the `azure` and `python` tags.

## Data Collection & Telemetry

The software collects information sent to Microsoft to improve services. Review Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default. To opt out, disable it during client creation with a `NoUserAgentPolicy` (see example in original README).

## Reporting Security Issues

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.  All contributions require a Contributor License Agreement (CLA).  This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).