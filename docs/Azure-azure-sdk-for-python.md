# Azure SDK for Python: Build Powerful Python Applications with Azure Services

**The Azure SDK for Python provides a comprehensive set of libraries to help you seamlessly integrate your Python applications with Azure services, enabling you to build robust, scalable, and secure cloud solutions.**  ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

## Key Features:

*   **Broad Service Coverage:** Access a wide array of Azure services, including compute, storage, networking, databases, and more.
*   **Modern and Consistent APIs:** Leverage well-designed, easy-to-use APIs that follow Azure SDK design guidelines for a consistent developer experience.
*   **Production-Ready Libraries:** Utilize stable, production-ready client and management libraries for reliable and performant application development.
*   **Authentication:** Utilize Azure Identity library for easy and secure authentication.
*   **Advanced Functionality:** Enjoy features like retry mechanisms, logging, transport protocols, and authentication protocols.
*   **Management Capabilities:** Provision and manage Azure resources effectively with comprehensive management libraries.

## Getting Started

To begin using the Azure SDK for Python, choose a service library from the `/sdk` directory. Detailed instructions can be found in the `README.md` or `README.rst` file within each library's project folder.

### Prerequisites

The client libraries support Python 3.9 or later. See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more information.

## Available Packages

The Azure SDK for Python offers various packages categorized for different services and release types:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

Explore the latest **General Availability (GA)** and **preview** client libraries, designed for consuming and interacting with existing Azure resources (e.g., uploading blobs). These libraries share core functionalities defined in the `azure-core` library. For the latest package list, visit the [latest releases page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Note:** Ensure your code is production-ready by using stable, non-preview libraries.

### Client: Previous Versions

Use the stable, production-ready client libraries for previous versions of Azure services. These may offer wider service coverage.

### Management: New Releases

Utilize the new management libraries built following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), including the Azure Identity library, an HTTP Pipeline, and error handling. Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt). Migrate from older versions with the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). For the most up-to-date package list, visit the [latest management releases page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

>   **Note:** Use stable, non-preview libraries for production. Refer to the migration guide if you encounter authentication issues after upgrading.

### Management: Previous Versions

For management libraries to provision and manage Azure resources, check [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries may have wider coverage than the new releases. Management libraries use namespaces starting with `azure-mgmt-`, e.g., `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issue Tracking:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community Support:** [Stack Overflow](https://stackoverflow.com/questions/tagged/azure+python) (use `azure` and `python` tags)

## Data Collection

The software collects telemetry data that is sent to Microsoft. You can turn this off. Learn more in the help documentation and Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704), and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is on by default. To opt-out, disable telemetry during client construction:

1.  Create a `NoUserAgentPolicy` class subclassing `UserAgentPolicy` with an empty `on_request` method.
2.  Pass an instance of this class as `user_agent_policy=NoUserAgentPolicy()` during client creation.

Example (using `azure-storage-blob`):

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

## Contributing

Contribute by following the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) and agreeing to the Contributor License Agreement (CLA). Learn more at https://cla.microsoft.com.

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.