# Azure SDK for Python: Simplify Cloud Development

Develop robust Python applications for the Azure cloud with the official Azure SDK for Python, providing comprehensive libraries for various Azure services.  [View the original repository](https://github.com/Azure/azure-sdk-for-python).

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

**Key Features:**

*   **Comprehensive Library Coverage:** Access a wide range of Azure services through dedicated Python libraries.
*   **Modular Package Structure:** Choose and use only the specific libraries you need, avoiding large, monolithic packages.
*   **Up-to-Date Releases:** Stay current with the latest features and improvements with both new and previous versions available.
*   **Management Libraries:** Provision and manage Azure resources using the latest management libraries.
*   **Azure SDK Design Guidelines:** Consistent API design across libraries for a familiar and intuitive experience.
*   **Production-Ready:** Benefit from stable, non-preview libraries for production deployments.
*   **Telemetry Configuration:** Opt-out options for controlling data collection.

## Getting Started

The Azure SDK for Python provides separate libraries for each Azure service.  Begin by selecting the library for the specific service you want to use. Refer to the `README.md` or `README.rst` files within each library's project folder for detailed instructions.

### Prerequisites

*   Python 3.9 or later
*   [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy)

## Available Packages

The SDK offers libraries categorized into:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These packages provide GA (Generally Available) and preview access to existing Azure resources. They share core functionalities like retry mechanisms, logging, authentication protocols, and more, outlined in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. For design guidelines, refer to [here](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production use, opt for stable, non-preview libraries.

### Client: Previous Versions

These are stable versions of packages for production use. While they may not follow the latest guidelines or have the same feature set as the new releases, they offer broader service coverage.

### Management: New Releases

Management libraries, compliant with [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), enable provisioning and management of Azure resources. Includes features like Azure Identity library, HTTP Pipeline, error-handling, and tracing.

Documentation and samples are available [here](https://aka.ms/azsdk/python/mgmt). A migration guide can be found [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** For production, use stable libraries.  Review the migration guide if you encounter authentication issues after upgrading management packages.

### Management: Previous Versions

For a list of management libraries, see [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries are identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow: Use tags `azure` and `python`.

## Data Collection

The SDK collects usage information for service and product improvement. [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) provide more information. Review Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).

### Telemetry Configuration

Telemetry is enabled by default.  To opt-out, use a `NoUserAgentPolicy` during client creation.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). This project welcomes contributions, which require a Contributor License Agreement (CLA).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.