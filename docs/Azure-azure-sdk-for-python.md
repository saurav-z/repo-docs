# Azure SDK for Python: Simplify Cloud Development

Easily build and manage applications on Azure using the official Azure SDK for Python. ([View the original repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, databases, and more.
*   **Client and Management Libraries:** Utilize both client libraries for interacting with existing resources and management libraries for provisioning and managing Azure resources.
*   **Modern Design Guidelines:** Benefit from libraries adhering to the latest Azure SDK design guidelines for Python, promoting consistency, and ease of use.
*   **Simplified Authentication:** Leverage the intuitive Azure Identity library for secure and streamlined authentication.
*   **Robust Core Functionality:** Take advantage of shared core functionalities like retries, logging, and authentication protocols, provided by the `azure-core` library.
*   **Production-Ready Packages:** Utilize stable, non-preview libraries for production environments, ensuring reliability and stability.

## Getting Started

The Azure SDK for Python offers separate libraries for each service. To begin, explore the `README.md` (or `README.rst`) file within each service's project folder, found in the `/sdk` directory.

### Prerequisites

The client libraries support Python 3.9 or later.  Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Package Categories

Explore the available packages categorized as follows:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These packages are generally available (GA) or in preview, enabling interaction with existing resources (e.g., uploading a blob). They share core functionalities like retries and authentication from the `azure-core` library.  Read the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) for best practices.  Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** Use stable, non-preview libraries for production deployments.

### Client: Previous Versions

These are the latest stable, production-ready packages, offering similar functionality to the preview releases, though they might not implement all guidelines.

### Management: New Releases

These libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and provide features like the Azure Identity library, an HTTP pipeline, and more.  Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).  Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** Use stable, non-preview libraries for production. Review the migration guide if you experience authentication issues after upgrading.

### Management: Previous Versions

Find a list of these libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). They may not have all the features of the new releases.  Management libraries begin with `azure-mgmt-`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   [StackOverflow](https://stackoverflow.com/questions/tagged/azure+python) (use tags `azure` and `python`)

## Data Collection and Telemetry

The SDK collects data for service improvement.  You can opt out of telemetry.

### Telemetry Configuration

Telemetry is on by default.  To disable it, override the `UserAgentPolicy` as shown in the example.

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

Report security issues to the Microsoft Security Response Center (MSRC) via <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.  Contributions require a Contributor License Agreement (CLA).

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.