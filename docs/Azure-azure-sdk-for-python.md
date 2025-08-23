# Azure SDK for Python: Simplify Cloud Development with Powerful Python Libraries

The Azure SDK for Python provides a comprehensive set of libraries to help you build, deploy, and manage applications on Microsoft Azure.  [Explore the Azure SDK for Python on GitHub](https://github.com/Azure/azure-sdk-for-python).

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Modern Design:** Leverage libraries built following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), ensuring a consistent and intuitive developer experience.
*   **Client and Management Libraries:** Utilize both client libraries for interacting with existing resources (e.g., uploading a blob) and management libraries for provisioning and managing Azure resources.
*   **Robust Authentication:** Benefit from the intuitive Azure Identity library for secure and streamlined authentication.
*   **Up-to-Date Packages:** Access the latest releases and preview features, with options for production-ready stable libraries.
*   **Core Functionality:** Shared functionalities like retries, logging, and authentication are available in the core libraries like `azure-core`.
*   **Detailed Documentation:** Find comprehensive documentation, code samples, and migration guides to get started quickly.

## Getting Started

Start developing on Azure using Python. Find a specific library by exploring the `/sdk` directory, or check out the public developer docs for detailed guidance.

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)

### Prerequisites

The client libraries support Python 3.9 or later. For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages:

Azure services are accessible through libraries categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries are **GA** or in **preview**, allowing you to interact with existing Azure resources (e.g., upload a blob).  They share core functionalities found in `azure-core`.  Find the [most up-to-date list of all of the new packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** Use stable, non-preview libraries for production.

### Client: Previous Versions

Stable, production-ready versions of packages offering similar functionality to the new releases. These provide wider coverage of Azure services.

### Management: New Releases

These libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and offer shared capabilities like Azure Identity and error-handling.

*   Documentation and code samples: [here](https://aka.ms/azsdk/python/mgmt).
*   Migration guide: [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).
*   [Up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** Use stable, non-preview libraries for production. If you upgraded to the new SDK versions, follow the migration guide mentioned above for authentication code changes.

### Management: Previous Versions

Libraries enabling you to provision and manage Azure resources.

*   [Complete list of management libraries](https://azure.github.io/azure-sdk/releases/latest/all/python.html).
*   Management libraries are identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   File an issue via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Check [previous questions](https://stackoverflow.com/questions/tagged/azure+python) or ask new ones on StackOverflow using `azure` and `python` tags.

## Data Collection

The software may collect information about your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described below. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

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

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) <secure@microsoft.com>. You should receive a response within 24 hours. Further information, including the MSRC PGP key, can be found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.

This project welcomes contributions under the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.