# Azure SDK for Python: Simplify Your Cloud Development

**Build robust and scalable applications on Microsoft Azure with the official Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository houses the actively developed Azure SDK for Python, providing a comprehensive suite of libraries to interact with Azure services.

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services through dedicated libraries.
*   **Modern Design:** Leverage libraries built on the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for consistent and intuitive APIs.
*   **Client and Management Libraries:** Utilize both client libraries for interacting with existing resources and management libraries for provisioning and managing Azure resources.
*   **Shared Core Functionality:** Benefit from core features like retries, logging, authentication, and more, shared across libraries.
*   **Production-Ready:** Access stable, non-preview libraries for production environments.
*   **Version Support:** The client libraries are supported on Python 3.9 or later.

## Getting Started

To use a specific service, select a library from the `/sdk` directory. Each library's `README.md` (or `README.rst`) provides detailed instructions.

### Prerequisites

*   Python 3.9 or later.

## Available Packages

Libraries are categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA** and **preview** libraries to use existing resources, like uploading blobs. These share common features like retries and logging. You can find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

> **Note:** Use stable, non-preview libraries for production.

### Client: Previous Versions

These are the last stable versions, production-ready, offering similar functionalities to the preview ones.

### Management: New Releases

New management libraries built following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). These provide core capabilities like Azure Identity, an HTTP Pipeline, error handling, and tracing. Documentation and samples are [here](https://aka.ms/azsdk/python/mgmt).  A migration guide is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

You can find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Note:** Use stable, non-preview libraries for production.  Refer to the migration guide if you experience authentication issues after upgrading.

### Management: Previous Versions

These provide a complete list of management libraries to provision and manage Azure resources.  See [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html) for a complete list. Management libraries are typically identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Search or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection & Telemetry

The software collects usage information and sends it to Microsoft for service improvement. You can disable telemetry by implementing a `NoUserAgentPolicy`. For more information, see the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

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

Report security vulnerabilities privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  You should receive a response within 24 hours.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing. All contributions require a Contributor License Agreement (CLA).

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.