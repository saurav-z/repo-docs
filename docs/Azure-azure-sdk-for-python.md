# Azure SDK for Python: Simplify Cloud Development with Python

**Build robust and scalable Python applications for the cloud with the official Azure SDK for Python!**  This library provides a comprehensive suite of packages to interact with various Azure services.  Get started with the [Azure SDK for Python](https://github.com/Azure/azure-sdk-for-python).

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide range of Azure services including storage, compute, networking, databases, and more.
*   **Modern API Design:** Leverage intuitive and Pythonic APIs for easy integration and development.
*   **Client and Management Libraries:** Choose from client libraries for interacting with existing resources and management libraries for provisioning and managing resources.
*   **Robust Authentication:** Simplify authentication with Azure Active Directory and other authentication options.
*   **Consistent Experience:** Benefit from shared core functionalities like retries, logging, and authentication across all libraries.
*   **Up-to-Date Documentation:** Access comprehensive documentation and code samples to accelerate your development process.
*   **Telemetry and Data Collection:** Understand data collection practices and how to opt-out.
*   **Easy Contribution:** Contribute to the project via the contributing guide.

## Getting Started

The Azure SDK for Python offers separate libraries for each service. Explore the `/sdk` directory to find the service libraries.

**Prerequisites:**
*   Python 3.9 or later.

## Packages Available

The SDK provides various packages, categorized for your convenience:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

Interact with existing Azure resources through **GA** (Generally Available) and **preview** packages. These libraries offer core functionalities like retries, logging, and authentication.  Find the [most up-to-date list of new packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** Use stable, non-preview libraries for production environments.

### Client: Previous Versions

Stable versions of packages that are production-ready. They provide similar functionalities to the Preview ones with a wider coverage of services.

### Management: New Releases

Provision and manage Azure resources using the management libraries, adhering to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). Access the [latest list of management packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html) and documentation [here](https://aka.ms/azsdk/python/mgmt). Migration guides are available for updating from older versions [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

> **Note:** If experiencing authentication issues after upgrading management libraries, refer to the migration guide.

### Management: Previous Versions

Explore the full list of management libraries for resource management [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries are identified by namespaces starting with `azure-mgmt-`.

## Need Help?

*   Detailed documentation: [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   File an issue: [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Ask questions on StackOverflow using `azure` and `python` tags.

## Data Collection

The software collects information about you and your use of the software and sends it to Microsoft. Learn more about data collection and the Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). See the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) for more details.

### Telemetry Configuration

Telemetry collection is on by default.

To opt out, disable telemetry during client construction:

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

## Reporting Security Issues and Security Bugs

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).