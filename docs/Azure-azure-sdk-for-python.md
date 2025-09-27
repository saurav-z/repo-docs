# Azure SDK for Python: Build Powerful Python Applications with Azure Services

**Supercharge your Python projects by seamlessly integrating with a wide array of Azure services using the official Azure SDK for Python.**

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository hosts the actively developed **Azure SDK for Python**, providing a comprehensive set of libraries to interact with Azure services. Visit the [official documentation](https://docs.microsoft.com/python/azure/) for detailed guidance.

## Key Features:

*   **Comprehensive Service Coverage:** Access a vast range of Azure services, including storage, compute, networking, databases, and more.
*   **Modern Design & SDK Guidelines:** Adheres to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for a consistent and intuitive developer experience.
*   **Client and Management Libraries:** Offers separate libraries for both client-side interaction (e.g., uploading blobs) and resource management (e.g., provisioning virtual machines).
*   **Production-Ready Libraries:** Utilize stable, non-preview libraries for production environments.
*   **Robust Core Functionality:** Benefit from shared features like retries, logging, authentication protocols, and tracing.
*   **Up-to-date Packages**: Find the latest packages on the [packages page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

## Getting Started

Choose from individual service libraries for a streamlined experience. Find the `README.md` (or `README.rst`) within each library's project folder for specific instructions. Service libraries are located in the `/sdk` directory.

### Prerequisites

The client libraries support Python 3.9 or later. For more information, see the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The Azure SDK for Python offers packages categorized into these main areas:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries provide access to existing resources and enable interactions like uploading blobs. They share core functionalities found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library and follow the defined [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [latest package information here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production code, use the stable, non-preview libraries.

### Client: Previous Versions

These are the stable, production-ready versions of packages that are used to work with Azure resources. They provide similar functionality to the preview releases.

### Management: New Releases

These libraries are built using the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They provide shared capabilities, including the Azure Identity library, an HTTP Pipeline, and error handling.

Documentation and code samples for these new libraries can be found [here](https://aka.ms/azsdk/python/mgmt). A migration guide is also available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

The [latest list of new packages is here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** Use stable libraries for production. Review the migration guide if you experience authentication issues after upgrading management packages.

### Management: Previous Versions

For a list of management libraries, visit [this page](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries offer extensive service coverage.  They are identified by namespaces starting with `azure-mgmt-`, e.g., `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search with `azure` and `python` tags.

## Data Collection

The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described below. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

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

Report security issues privately via email to the [Microsoft Security Response Center (MSRC)](mailto:secure@microsoft.com).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing. This project welcomes contributions and follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

[Back to the main repository](https://github.com/Azure/azure-sdk-for-python)