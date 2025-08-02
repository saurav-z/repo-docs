# Azure SDK for Python: Simplify Cloud Development

Easily build and manage applications on Microsoft Azure with the **Azure SDK for Python**, offering a comprehensive set of libraries to interact with various Azure services.  [View the original repository](https://github.com/Azure/azure-sdk-for-python).

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide range of Azure services through dedicated Python libraries.
*   **Simplified Development:**  Benefit from intuitive APIs designed for ease of use and a consistent experience.
*   **Production-Ready Libraries:** Utilize stable, well-tested libraries for reliable production deployments.
*   **Azure SDK Design Guidelines:** Libraries following the Azure SDK Design Guidelines for Python, ensuring a consistent development experience.
*   **Management Libraries:** Manage and provision Azure resources using libraries adhering to Azure SDK guidelines.
*   **Active Development:** Benefit from ongoing improvements and updates to support the latest Azure features.
*   **Telemetry and Configuration:** Allows for the optional configuration of telemetry collection.

## Getting Started

Individual service libraries are available, allowing you to choose the specific packages you need. Refer to the `README.md` (or `README.rst`) files within each library's project folder to get started with a specific service.

## Prerequisites

The client libraries are supported on Python 3.9 or later.  See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Available Packages

Libraries are organized into categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries allow you to use and consume existing resources and interact with them, for example: upload a blob.  These libraries share core functionalities such as: retries, logging, transport protocols, authentication protocols, etc. that can be found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.  Learn more about these libraries by reading guidelines that they follow [here](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [most up-to-date list of all of the new packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

> **Note:** Use stable, non-preview libraries for production.

### Client: Previous Versions

These are the last stable versions of production-ready packages.  They may not have the same features or follow the same guidelines as the new releases, but offer wider service coverage.

### Management: New Releases

A new set of management libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). These new libraries provide core capabilities shared across all Azure SDKs.

Documentation and code samples can be found [here](https://aka.ms/azsdk/python/mgmt). A migration guide is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [most up to date list of all of the new packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Note:** Use stable, non-preview libraries for production. Refer to the migration guide if you experience authentication issues after upgrading management packages.

### Management: Previous Versions

For a complete list of management libraries that enable you to provision and manage Azure resources, please [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). They might not have the same feature set as the new releases but they do offer wider coverage of services.
Management libraries can be identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   [StackOverflow](https://stackoverflow.com/questions/tagged/azure+python) (using `azure` and `python` tags)

## Data Collection

The software may collect data that is sent to Microsoft. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry collection is on by default.

To opt out, disable telemetry during client construction.  Create a `NoUserAgentPolicy` class and pass an instance during client creation.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.

This project welcomes contributions, which require a Contributor License Agreement (CLA).  See https://cla.microsoft.com for details.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).