# Azure SDK for Python: Simplify Cloud Development

**Build robust and scalable Python applications with the official Azure SDK, providing comprehensive libraries for interacting with Azure services.**

[Link to Original Repo: Azure SDK for Python](https://github.com/Azure/azure-sdk-for-python)

## Key Features

*   **Comprehensive Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modular Design:** Leverage individual service libraries for a streamlined development experience, or use the large Azure package.
*   **Up-to-Date Packages:** Stay current with the latest features and improvements through new releases.
*   **Backward Compatibility:** Utilize previous versions of libraries for production-ready stability and wider service coverage.
*   **Management Libraries:** Provision and manage Azure resources with the new management libraries that follow the Azure SDK Design Guidelines for Python.
*   **Azure SDK Design Guidelines:** Packages are built following design guidelines for Python, ensuring a consistent and intuitive developer experience.
*   **Authentication Support:** Utilize a range of authentication methods, including Azure Active Directory and managed identities.
*   **Easy to Get Started:** Extensive documentation and code samples to help you get started quickly.
*   **Community Support:** Get help through documentation, GitHub Issues, and Stack Overflow.

## Getting Started

Choose from a range of service-specific libraries located in the `/sdk` directory. To get started with a specific library, refer to its `README.md` or `README.rst` file within its project folder.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For more details, please read our [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Package Categories

Each service provides libraries categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

Explore the latest **GA** (Generally Available) and **preview** libraries. These allow you to use and consume existing Azure resources. These libraries share core functionalities. You can find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

> NOTE: Utilize stable, non-preview libraries for production environments.

### Client: Previous Versions

Use last stable versions for production-ready usage with Azure. They provide functionalities and wider coverage of services.

### Management: New Releases

New libraries are available that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) are now available. Documentation and code samples for these new libraries can be found [here](https://aka.ms/azsdk/python/mgmt). In addition, a migration guide that shows how to transition from older versions of libraries is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

You can find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> NOTE: Ensure you use stable, non-preview libraries in production. Refer to the migration guide if upgrading management packages.

### Management: Previous Versions

Access the management libraries that enable you to provision and manage Azure resources, please [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). They might not have the same feature set as the new releases but they do offer wider coverage of services.

## Need Help?

*   Detailed documentation: [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   File an issue: [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Search Stack Overflow: Use tags `azure` and `python`.

## Data Collection & Telemetry

The SDK collects data to improve products and services. You can opt-out by disabling telemetry at client construction, by using the `NoUserAgentPolicy`.

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

### Reporting Security Issues

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details. This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).