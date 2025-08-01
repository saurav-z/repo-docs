# Azure SDK for Python: Simplify Cloud Development with Python

**(Link back to original repo: [https://github.com/Azure/azure-sdk-for-python](https://github.com/Azure/azure-sdk-for-python))**

This repository is the central hub for the actively developed Azure SDK for Python, providing comprehensive libraries to interact with Azure services.

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services with dedicated Python libraries.
*   **Modular Design:** Choose specific libraries for individual services instead of a single, monolithic package.
*   **Latest Releases:** Get access to the newest packages, including both GA (Generally Available) and preview versions.
*   **Production-Ready Libraries:** Utilize stable, non-preview libraries for production environments.
*   **Consistent Design Guidelines:** Benefit from libraries that adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), ensuring a consistent and intuitive developer experience.
*   **Management Libraries:** Manage and provision Azure resources using dedicated management libraries.
*   **Azure Identity Integration:** Leveraging built-in support for Azure Identity library to easily authenticate with Azure services.
*   **Detailed Documentation:** Access comprehensive documentation, including developer docs, migration guides and code samples.
*   **Active Community Support:** Get help through GitHub Issues, StackOverflow, and direct support from Microsoft.

## Getting Started

To begin, explore the service-specific `README.md` (or `README.rst`) files within each library's project folder in the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later. See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.

## Package Categories

Explore available packages in the following categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the newest Generally Available (GA) and preview libraries for interacting with Azure services, offering features like retries, logging, authentication, and transport protocols. Stay up-to-date with the [latest packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Note:** Use stable, non-preview libraries for production applications.

### Client: Previous Versions

These are the last stable versions of the client libraries and are production ready. They offer a broad coverage of Azure services.

### Management: New Releases

Discover new management libraries aligned with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). Find documentation and examples [here](https://aka.ms/azsdk/python/mgmt), and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

The [most up-to-date list of all of the new packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html) can be found here.

>   **Note:** If you need to ensure your code is ready for production use one of the stable, non-preview libraries. Also, if you are experiencing authentication issues with the management libraries after upgrading certain packages, it's possible that you upgraded to the new versions of SDK without changing the authentication code, please refer to the migration guide mentioned above for proper instructions.

### Management: Previous Versions

For a comprehensive list of management libraries to provision and manage Azure resources, see [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These can be identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **StackOverflow:** Search with `azure` and `python` tags ([previous questions](https://stackoverflow.com/questions/tagged/azure+python))

## Data Collection

The SDK collects telemetry data. More information can be found in the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default. To opt-out, create a `NoUserAgentPolicy` and pass it during client creation:

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

Report security issues to the Microsoft Security Response Center (MSRC) via <secure@microsoft.com>. See the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue) for more information.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.