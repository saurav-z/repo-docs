# Azure SDK for Python: Simplify Cloud Development

**Empower your Python applications with seamless integration to Azure cloud services using the official Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Modern API Design:** Leverage intuitive and consistent APIs designed for ease of use and developer productivity.
*   **Cross-Platform Compatibility:** Develop and deploy your Python applications on various platforms.
*   **Authentication & Security:** Benefit from robust authentication mechanisms and security best practices.
*   **Regular Updates:** Stay current with the latest Azure features and improvements.
*   **Flexible Package Structure:** Choose from individual service libraries for optimized dependencies.
*   **Well-documented:** Detailed documentation and code samples to get you started quickly.
*   **Telemetry Configuration:** Customize telemetry settings to manage data collection.
*   **Active Community:** Get help from the Azure community to solve any problems.

## Getting Started

The Azure SDK for Python offers modular libraries for specific Azure services. This approach enables you to include only the dependencies you need, keeping your project lightweight.

### Prerequisites

*   Python 3.9 or later.
*   Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

### Available Packages

Explore the following categories for service libraries:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

#### Client: New Releases

These are the latest generally available (GA) and preview libraries for interacting with Azure resources, offering functionalities like blob uploads and downloads.

[Up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

> **Note:** For production environments, use stable, non-preview libraries.

#### Client: Previous Versions

Previous stable releases, providing production-ready functionality. These may not fully implement the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) but offer wider service coverage.

#### Management: New Releases

New libraries, aligned with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), offer core capabilities like Azure Identity and error handling.

*   Documentation and samples: [here](https://aka.ms/azsdk/python/mgmt)
*   Migration guide: [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide)
*   [Up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Note:** Upgrade to the new SDK version while adapting authentication codes.

#### Management: Previous Versions

Libraries for provisioning and managing Azure resources.

*   Complete list: [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html)
*   Libraries identified by `azure-mgmt-` namespaces (e.g., `azure-mgmt-compute`).

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Stack Overflow ([azure](https://stackoverflow.com/questions/tagged/azure) and [python](https://stackoverflow.com/questions/tagged/python) tags)

## Data Collection

The SDK collects usage data to improve products and services. You can disable telemetry during client creation.

*   [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy)

### Telemetry Configuration

Disable telemetry by using a custom `NoUserAgentPolicy`:

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

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

*   [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue)

## Contributing

Contributions are welcome! See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).