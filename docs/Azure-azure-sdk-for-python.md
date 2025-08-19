# Azure SDK for Python: Build Powerful Applications on Azure

**Empower your Python applications with seamless integration and robust features using the official Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

The Azure SDK for Python provides a comprehensive set of libraries, enabling developers to easily access and manage Azure services. This empowers developers with a wealth of tools to build, deploy, and manage their applications on the Microsoft Azure cloud platform.

## Key Features

*   **Broad Service Coverage:** Access a wide range of Azure services, from storage and compute to databases and AI.
*   **Modern Design:** Follows the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for consistency and ease of use.
*   **Client & Management Libraries:** Choose between client libraries for interacting with resources and management libraries for provisioning and configuration.
*   **Simplified Authentication:** Leverage the intuitive Azure Identity library for secure and simplified authentication.
*   **Built-in Core Functionality:** Utilize shared core functionalities, such as retries, logging, transport and authentication protocols found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Production Ready:** Benefit from stable, non-preview libraries for production environments.
*   **Telemetry Configuration:** Easily opt-out of telemetry collection for increased privacy.

## Getting Started

*   **Prerequisites:**  Python 3.9 or later.  See the [version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.
*   **Individual Libraries:** Each service has its own set of libraries. Refer to the `README.md` (or `README.rst`) in each service's project folder within the `/sdk` directory.
*   **Developer Documentation:** Access the official documentation at [https://docs.microsoft.com/python/azure/](https://docs.microsoft.com/python/azure/).

## Packages Available

The Azure SDK for Python organizes libraries into categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA (Generally Available)** and **preview** packages designed for using and consuming existing resources. They incorporate core functionalities for efficient operations. Find the [most up-to-date list of new packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production, use stable, non-preview libraries.

### Client: Previous Versions

These are the stable, production-ready versions of the libraries that provide the same functionalities as the preview releases.

### Management: New Releases

These management libraries are built in accordance to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and offer new features. Find the [most up to date list of new packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** If you upgrade and experience authentication issues, refer to the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for proper instructions.

### Management: Previous Versions

These provide management capabilities for provisioning and managing Azure resources. A complete list can be [found here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries are identifiable by namespaces beginning with `azure-mgmt-`, such as `azure-mgmt-compute`.

## Need Help?

*   **Documentation:**  [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:**  [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search with `azure` and `python` tags.

## Data Collection and Telemetry

The SDK collects data to improve services (refer to the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy)).

### Telemetry Configuration

Telemetry is on by default. Opt-out by disabling it during client creation:

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

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

*   See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md).
*   All contributions require a Contributor License Agreement (CLA).
*   This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).