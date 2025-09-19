# Azure SDK for Python: Simplify Cloud Development with Python

**Simplify your Python cloud development with the official Azure SDK for Python, providing comprehensive libraries for seamless integration with Azure services.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services through dedicated Python libraries.
*   **Client and Management Libraries:** Choose from new and previous versions of client libraries to interact with existing resources and new management libraries to provision and manage Azure resources.
*   **Modern Design Guidelines:** Follows [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for consistency and ease of use.
*   **Core Functionality:** Shared core features such as retries, logging, authentication, and more through the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Production-Ready Stability:** Leverage stable, non-preview libraries for production deployments.
*   **Detailed Documentation:** Access comprehensive documentation at [Azure SDK for Python documentation](https://aka.ms/python-docs).

## Getting Started

The Azure SDK for Python provides separate libraries for each Azure service, allowing you to choose the specific components you need.

### Prerequisites

The client libraries are supported on Python 3.9 or later.

For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

### Service Libraries

Service libraries are available in the `/sdk` directory.  Each service has its own `README.md` (or `README.rst`) file within its project folder for specific getting started instructions.

## Package Categories

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

New releases of packages that are **GA** (Generally Available) and **preview**. These libraries provide access to existing resources and their functionalities.

[Most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

> **Note:** For production use, use stable, non-preview libraries.

### Client: Previous Versions

Stable, production-ready versions of packages for interacting with Azure services.  These offer wider service coverage.

### Management: New Releases

New management libraries built following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).

Documentation and code samples for these new libraries can be found [here](https://aka.ms/azsdk/python/mgmt).
A migration guide is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

[Most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Note:**  If experiencing authentication issues after upgrading, refer to the migration guide.

### Management: Previous Versions

Complete list of management libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).
Management libraries are identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow using `azure` and `python` tags.

## Data Collection

This software collects data which is sent to Microsoft. You may turn off the telemetry.  Learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).  For information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry collection is on by default.

To opt out, you can disable telemetry at client construction. Define a `NoUserAgentPolicy` class that is a subclass of `UserAgentPolicy` with an `on_request` method that does nothing. Then pass instance of this class as kwargs `user_agent_policy=NoUserAgentPolicy()` during client creation. This will disable telemetry for all methods in the client. Do this for every new client.

The example below uses the `azure-storage-blob` package. In your code, you can replace `azure-storage-blob` with the package you are using.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) <secure@microsoft.com>.  Further information can be found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).