# Azure SDK for Python: Build Powerful Python Applications on Azure

**Unlock the full potential of Microsoft Azure with the Azure SDK for Python, your comprehensive toolkit for building robust, scalable, and cloud-native applications.**

[Original Repository](https://github.com/Azure/azure-sdk-for-python)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Client and Management Libraries:** Choose from client libraries for interacting with existing resources (e.g., uploading blobs) and management libraries for provisioning and managing Azure resources.
*   **Production-Ready Libraries:** Leverage stable, production-ready libraries for reliable performance and compatibility.
*   **Modern Design Guidelines:** Benefit from libraries that adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), ensuring consistency and ease of use.
*   **Azure Identity Integration:** Simplify authentication with the intuitive Azure Identity library.
*   **Core Capabilities:** Benefit from shared core functionalities such as retries, logging, transport protocols, authentication protocols, etc. that can be found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.

## Getting Started

Find service-specific libraries in the `/sdk` directory. The libraries are supported on Python 3.9 or later. Read more about the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages

Explore libraries across these categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries offer access to existing Azure resources and interaction capabilities (e.g., uploading blobs). They share common features like retries, logging, and authentication.  Find the [latest packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** Use stable, non-preview libraries for production.

### Client: Previous Versions

Access stable versions of packages that provide similar functionalities to the Preview ones.

### Management: New Releases

These libraries enable provisioning and managing Azure resources, following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). Explore documentation and samples [here](https://aka.ms/azsdk/python/mgmt). Review the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for upgrading.
Find the [latest packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** Use stable, non-preview libraries for production.

### Management: Previous Versions

For a full list of management libraries to provision and manage Azure resources, check [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries can be identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow with `azure` and `python` tags.

## Data Collection

The software collects usage data. You can disable telemetry. Learn more in the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is on by default.  You can opt out using the following example:

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

Report security issues privately to the [Microsoft Security Response Center (MSRC)](mailto:secure@microsoft.com).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).