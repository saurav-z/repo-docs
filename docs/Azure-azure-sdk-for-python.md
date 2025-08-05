# Azure SDK for Python: Build Powerful Python Applications with Azure Services

**Simplify your cloud development experience with the Azure SDK for Python, offering a comprehensive suite of libraries to interact with Azure services.**  [Explore the Azure SDK for Python on GitHub](https://github.com/Azure/azure-sdk-for-python).

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services including storage, compute, databases, AI, and more.
*   **Modern Design:** Adheres to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) ensuring consistency and ease of use.
*   **Modular Libraries:** Utilize separate libraries for each service, allowing you to include only the dependencies you need.
*   **Robust Authentication:** Leverage the intuitive Azure Identity library for secure and simplified authentication.
*   **Production-Ready:** Choose stable, non-preview libraries for production environments.
*   **Consistent Experience:** Core functionalities like retries, logging, and authentication are shared across libraries, streamlining development.
*   **Management Capabilities:** Manage and provision Azure resources using dedicated management libraries.
*   **Detailed Documentation:** Comprehensive documentation is available on [Azure SDK for Python documentation](https://aka.ms/python-docs).

## Getting Started

To start using a specific Azure service, choose the appropriate library from the `/sdk` directory. Each library includes a `README.md` or `README.rst` file with detailed instructions.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The Azure SDK for Python offers a variety of libraries categorized by function and release status:

*   **Client - New Releases:**  Latest **GA** (Generally Available) and preview libraries for interacting with Azure resources (e.g., uploading blobs).  Find the [most up to date list](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
*   **Client - Previous Versions:** Stable versions of client libraries for production use.
*   **Management - New Releases:** Management libraries adhering to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for provisioning and managing Azure resources. See documentation and code samples [here](https://aka.ms/azsdk/python/mgmt) and the list of [latest packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).
*   **Management - Previous Versions:** Older management libraries for managing Azure resources.  See [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html) for a full list.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Search or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection

The Azure SDK may collect information about your use of the software for service improvement. You can control telemetry collection as detailed below and in the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy). See Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) for more details.

### Telemetry Configuration

Telemetry is enabled by default.

To disable telemetry:

1.  Create a subclass of `UserAgentPolicy` (e.g., `NoUserAgentPolicy`) and override the `on_request` method to do nothing.
2.  Pass an instance of this class as the `user_agent_policy` keyword argument when creating client objects. This will disable telemetry for all methods in the client.

**Example (using azure-storage-blob):**

```python
import os
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.core.pipeline.policies import UserAgentPolicy

# Create your credential
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

Report security vulnerabilities privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.  All contributions require a Contributor License Agreement (CLA).

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).