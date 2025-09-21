# Azure SDK for Python: Simplify Cloud Development in Python

**Easily build, deploy, and manage applications on Microsoft Azure using the comprehensive Azure SDK for Python.** This repository provides the source code and resources for the official Azure SDK for Python, enabling developers to interact with various Azure services efficiently. ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, databases, and more.
*   **Well-Documented and Consistent APIs:** Benefit from clear documentation and a consistent API design across all Azure services.
*   **Modern Authentication:** Seamlessly integrate with Azure Active Directory and other authentication methods.
*   **Production-Ready Libraries:** Choose from stable, non-preview libraries for reliable production deployments.
*   **Regular Updates:** Stay up-to-date with the latest features and improvements from Azure.
*   **Management Libraries**: Manage and provision Azure resources programmatically.

## Getting Started

Each Azure service is available as a separate Python library. Find the `README.md` (or `README.rst`) within each library's project folder in the `/sdk` directory for specific installation and usage instructions.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For more details, please read our [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The Azure SDK for Python offers packages categorized into:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA** (Generally Available) and **preview** packages for interacting with Azure resources, like uploading a blob. These share core functionalities like retries, logging, and authentication found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.  Find the most up-to-date list [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Important:** Use stable, non-preview libraries for production environments.

### Client: Previous Versions

Stable, production-ready versions of packages. These provide similar functionalities to the newer releases and are suitable for production use.

### Management: New Releases

New management libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).  These offer core capabilities, including Azure Identity, HTTP Pipeline with custom policies, error handling, and distributed tracing. Documentation and samples are available [here](https://aka.ms/azsdk/python/mgmt). A migration guide is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). Find the most up-to-date list [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

>   **Important:** For production, use stable, non-preview libraries. If you're experiencing authentication issues after upgrading, consult the migration guide.

### Management: Previous Versions

For a complete list of management libraries, see [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries, identified by `azure-mgmt-` namespaces (e.g., `azure-mgmt-compute`), may offer wider service coverage.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow: Use tags `azure` and `python` to search or ask questions.

## Data Collection and Telemetry

The SDK collects usage data to improve services. See the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) and Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) for details.

### Telemetry Configuration

Telemetry is enabled by default. To disable it, use the `NoUserAgentPolicy` class when creating clients.

```python
import os
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.core.pipeline.policies import UserAgentPolicy

# ... (rest of the code) ...

class NoUserAgentPolicy(UserAgentPolicy):
    def on_request(self, request):
        pass

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient(account_url, credential=mi_credential, user_agent_policy=NoUserAgentPolicy())
```

## Reporting Security Issues

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours.

## Contributing

Contribute to the project by following the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). All contributions require a Contributor License Agreement (CLA). This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).