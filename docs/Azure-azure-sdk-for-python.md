# Azure SDK for Python: Simplify Cloud Development with Powerful Python Libraries

**Empower your Python applications to seamlessly interact with Azure services using the official and comprehensive Azure SDK for Python.** [View the source code on GitHub](https://github.com/Azure/azure-sdk-for-python).

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide array of Azure services through dedicated Python libraries.
*   **Latest Releases & Preview Features:** Stay at the forefront with new and preview releases, including GA packages for production readiness.
*   **Management Libraries:** Provision and manage Azure resources with the latest management libraries, adhering to Azure SDK Design Guidelines.
*   **Robust Core Functionality:** Benefit from shared core functionalities like retries, logging, and authentication, provided by the `azure-core` library.
*   **Production-Ready Libraries:** Leverage stable, non-preview libraries for production environments.
*   **Detailed Documentation:** Access comprehensive documentation and samples to get started quickly.
*   **Flexible Telemetry:** Control telemetry collection to meet your application's needs.

## Getting Started

Each Azure service offers a dedicated Python library for easy integration. Find service-specific libraries within the `/sdk` directory, or explore the [public developer docs](https://docs.microsoft.com/python/azure/) for guidance.

### Prerequisites

The client libraries require Python 3.9 or later.  Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Package Categories

Explore available packages across these categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries provide access to Azure services and utilize shared core functionalities. Find the [most up-to-date package list](https://azure.github.io/azure-sdk/releases/latest/index.html#python) for new releases, including preview versions.  Ensure production readiness by using stable, non-preview libraries.

### Client: Previous Versions

These packages offer stable versions of client libraries for production use. They provide comparable functionality to the preview libraries, with wider service coverage.

### Management: New Releases

New management libraries are available, aligning with [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) offering core capabilities and an intuitive Azure Identity library. Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

The [latest packages are available here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

### Management: Previous Versions

Explore the full list of management libraries for provisioning and managing Azure resources [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries are identified by the `azure-mgmt-` prefix.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Search or ask on StackOverflow with the tags `azure` and `python`.

## Data Collection

This software collects usage data which is sent to Microsoft. Review Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) for more information.

### Telemetry Configuration

Telemetry collection is enabled by default. To opt-out, disable telemetry at client construction.

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

## Security

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  Find more details at the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).