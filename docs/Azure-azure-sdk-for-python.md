# Azure SDK for Python: Develop Robust Applications for the Cloud

The Azure SDK for Python provides a comprehensive set of libraries to interact with Azure services, empowering developers to build powerful and scalable cloud applications. ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Modular Libraries:** Utilize individual libraries for specific services, allowing you to choose only the components you need.
*   **Latest Releases & Previous Versions:** Access both the newest GA and preview releases, with the option of selecting previous stable versions.
*   **Management Libraries:** Provision and manage Azure resources with the new Management libraries.
*   **Azure SDK Design Guidelines:** Designed with the Azure SDK Design Guidelines for Python
*   **Cross-Platform Compatibility:** Supported on Python 3.9 and later.
*   **Built-in Features:** Including authentication, retries, logging, and telemetry for easy development.
*   **Detailed Documentation:** Access extensive documentation, code samples, and migration guides to accelerate development.
*   **Telemetry Control:** Easily control telemetry collection for your applications.

## Getting Started

Each service library is available as a separate package. To get started, find the `README.md` (or `README.rst`) in the library's project folder within the `/sdk` directory.

## Available Packages

The SDK offers various packages categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA** and **preview** packages. They allow you to use existing resources and interact with them, sharing core functionalities like retries, logging, and authentication. You can find the up-to-date list [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Note:** For production use, consider stable, non-preview libraries.

### Client: Previous Versions

These are the last stable, production-ready versions, offering similar functionalities. They may not fully adhere to the latest guidelines but provide wider service coverage.

### Management: New Releases

These libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and provide features such as intuitive Azure Identity, an HTTP Pipeline, error handling, and more. Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).
You can find the most up to date list of all of the new packages on our page [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

>   **Note:** For production, use stable libraries. Refer to the migration guide if you experience authentication issues after upgrading.

### Management: Previous Versions

Find a complete list of libraries for provisioning and managing Azure resources [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). They are identified by namespaces starting with `azure-mgmt-`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow (using `azure` and `python` tags)

## Data Collection

The SDK collects data to improve services. You can disable telemetry as shown below.
Find more about data collection on the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is on by default. To opt out, use the `NoUserAgentPolicy` when creating clients.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md).
This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).