# Azure SDK for Python

**Supercharge your Python applications with the official Azure SDK, offering comprehensive tools and libraries for seamless integration with Microsoft Azure services.**

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository provides the source code for the Azure SDK for Python. Explore comprehensive documentation and examples to quickly and efficiently integrate Azure services into your Python projects.

[**Visit the Original Repository**](https://github.com/Azure/azure-sdk-for-python)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, databases, AI, and more.
*   **Up-to-Date Libraries:** Stay current with the latest releases and features.
*   **Easy Integration:** Utilize well-documented and user-friendly libraries designed for Python developers.
*   **Production-Ready:** Leverage stable, non-preview libraries for production environments.
*   **Azure SDK Design Guidelines:** The Management libraries follow the Azure SDK Design Guidelines for Python.
*   **Robust Core Functionality:** Benefit from core features like retries, logging, authentication, and more, built into the `azure-core` library.
*   **Management and Client Libraries:** Leverage distinct libraries for managing and consuming Azure resources.

## Getting Started

Choose the appropriate libraries for the Azure services you're using.  Each service library has its own `README.md` (or `README.rst`) file within its project folder, providing detailed instructions.  Service libraries are located within the `/sdk` directory.

### Prerequisites

The client libraries support Python 3.9 or later. For details, review the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The Azure SDK offers service libraries in the following categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries offer the latest **GA** and **preview** releases, allowing you to interact with existing Azure resources. Core functionalities are shared and can be found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.

*   Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
    > **Important:**  Use stable, non-preview libraries for production deployments.

### Client: Previous Versions

Stable, production-ready versions of packages are available, offering a broad range of service support.

### Management: New Releases

These libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and provide core capabilities, including the intuitive Azure Identity library and an HTTP Pipeline.

*   Documentation and samples are available [here](https://aka.ms/azsdk/python/mgmt).
*   A migration guide is [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).
*   Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).
    > **Important:** Use stable, non-preview libraries for production deployments.  Refer to the migration guide if you encounter authentication issues after upgrading.

### Management: Previous Versions

Complete lists of management libraries (namespaces starting with `azure-mgmt-`) are [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

## Need Help?

*   Comprehensive documentation is available at [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Search or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection

The software collects information about you and your use of the software and sends it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described below. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default.

To disable it, create a custom `NoUserAgentPolicy` (subclass of `UserAgentPolicy`) that does nothing in the `on_request` method and pass it as `user_agent_policy=NoUserAgentPolicy()` during client construction.

Example:

```python
import os
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.core.pipeline.policies import UserAgentPolicy


# Create your credential you want to use
mi_credential = ManagedIdentityCredential()

account_url = "https://<storageaccountname>.blob.core.windows.net"

# Set up user-agent override
class NoUserAgentPolicy(UserAgentPolicy):
    def on_request(self, request):
        pass

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient(account_url, credential=mi_credential, user_agent_policy=NoUserAgentPolicy())

container_client = blob_service_client.get_container_client(container=<container_name>)
# TODO: do something with the container client like download blob to a file
```

### Reporting Security Issues

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. Further information, including the MSRC PGP key, can be found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.

This project welcomes contributions under the [Microsoft Contributor License Agreement (CLA)](https://cla.microsoft.com).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.