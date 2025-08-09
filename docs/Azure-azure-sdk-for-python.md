# Azure SDK for Python: Build Powerful Python Applications on Azure

**Empower your Python applications with the official Azure SDK, providing comprehensive tools and libraries for seamless integration with Azure services.**

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository hosts the active development of the Azure SDK for Python. For in-depth information and documentation, please visit the [public developer docs](https://docs.microsoft.com/python/azure/) or the versioned [developer docs](https://azure.github.io/azure-sdk-for-python).

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, databases, networking, and more.
*   **Client & Management Libraries:** Choose from client libraries for interacting with existing resources (e.g., uploading blobs) and management libraries for provisioning and managing Azure resources.
*   **Modern Design Guidelines:** Benefit from libraries built on the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), offering a consistent and intuitive development experience.
*   **Robust Core Functionality:** Leverage shared features like retries, logging, authentication, and transport protocols provided by the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Version Support:** Client libraries are supported on Python 3.9 or later.

## Getting Started

The Azure SDK for Python is organized into individual libraries for each Azure service. To get started:

1.  **Explore the SDK:** Browse the `/sdk` directory to find the service libraries.
2.  **Review the README:** Each service library includes a `README.md` (or `README.rst`) file with specific instructions for installation and usage.
3.  **Prerequisites:** Ensure you have Python 3.9 or later installed. See [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.

## Available Packages

The SDK provides packages categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

New generation of libraries that provide access to use and consume existing resources. They share core functionality and are built following the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html). Find the latest packages on our [release page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** Use stable, non-preview libraries for production environments.

### Client: Previous Versions

Stable versions of packages. While these might not implement all the latest guidelines or have the same feature set, they offer wider service coverage.

### Management: New Releases

New libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) designed for provisioning and managing Azure resources. Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt). Migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). Find the latest packages on our [release page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** Use stable, non-preview libraries for production environments. Refer to the migration guide if experiencing authentication issues after upgrading.

### Management: Previous Versions

These libraries enable you to provision and manage Azure resources. Find a complete list [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Identify management libraries by namespaces starting with `azure-mgmt-`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Search or ask questions on StackOverflow using the tags `azure` and `python`.

## Data Collection

The SDK collects telemetry data to improve the service. Learn more in the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) and the [Microsoft privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).

### Telemetry Configuration

Telemetry is enabled by default. To disable it, create a custom `NoUserAgentPolicy` to pass to your client.

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

## Reporting Security Issues

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.

## Additional Resources

*   [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
*   [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)

[Back to Top](#azure-sdk-for-python-build-powerful-python-applications-on-azure) - [Return to Original Repo](https://github.com/Azure/azure-sdk-for-python)