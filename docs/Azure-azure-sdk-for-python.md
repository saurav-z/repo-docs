# Azure SDK for Python: Simplify Cloud Development with Powerful Python Libraries

**Quickly and easily build and manage applications on Microsoft Azure using the official Azure SDK for Python.**  ([See the original repo](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Coverage:** Access a wide array of Azure services through dedicated Python libraries.
*   **Simplified Development:** Leverage libraries designed for ease of use and integration with Python best practices.
*   **Consistent Experience:** Benefit from shared core functionalities like retries, logging, and authentication across multiple services.
*   **Up-to-date Releases:** Stay current with the latest features and improvements, with both new releases and previous versions available.
*   **Management Capabilities:** Provision and manage Azure resources through the Management libraries.
*   **Robust Documentation:** Access detailed documentation and code samples to get started quickly.
*   **Telemetry Options:** Control data collection for privacy compliance, with opt-out options.

## Getting Started

Each Azure service has its own set of Python libraries. To get started, explore the `README.md` (or `README.rst`) file within the specific library's project folder within the `/sdk` directory.

### Prerequisites

The client libraries support Python 3.9 or later.  See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.

## Package Categories

The SDK offers libraries in the following categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA** (Generally Available) and preview releases for interacting with Azure resources (e.g., uploading blobs). These libraries are built with common core features such as retries, logging, and authentication, provided by the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.

Get the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production use, choose stable, non-preview libraries.

### Client: Previous Versions

These are the last stable versions of packages that provide functionality for Azure with wider coverage of services than the new releases, although they may not implement the same guidelines.

### Management: New Releases

These management libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and include the Azure Identity library, an HTTP Pipeline, error handling, and tracing.

Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Get the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Note:** For production, choose stable libraries.  If you upgraded certain packages and are experiencing authentication issues, consult the migration guide.

### Management: Previous Versions

Find a complete list of management libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries offer wider coverage of services, but may not have the same features as the new releases.  Management libraries use namespaces starting with `azure-mgmt-` (e.g., `azure-mgmt-compute`).

## Need Help?

*   Visit the [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   File an issue via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   Search or ask questions on StackOverflow with the `azure` and `python` tags.

## Data Collection and Telemetry

This software collects information about your usage and sends it to Microsoft.  You can learn more about this and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). See the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) for details.

### Telemetry Configuration

Telemetry is enabled by default.

To opt out, disable telemetry at client construction by passing in a `NoUserAgentPolicy` during client creation.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. See the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue) for more information.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.  All contributions require a Contributor License Agreement (CLA).

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.