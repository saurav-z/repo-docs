# Azure SDK for Python: Simplify Your Cloud Development

**Empower your Python applications with seamless integration and management of Azure services using the official Azure SDK for Python.**

[Link to Original Repo: Azure SDK for Python](https://github.com/Azure/azure-sdk-for-python)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide array of Azure services through dedicated libraries, covering everything from storage and compute to AI and databases.
*   **Modern Design:** Adheres to the latest Azure SDK Design Guidelines for Python, ensuring consistent and intuitive APIs, including Azure Identity library, HTTP Pipeline, custom policies, and error handling.
*   **Production-Ready Libraries:** Utilize stable, production-ready libraries for reliable performance and stability in your critical applications.
*   **Up-to-Date Releases:** Stay current with the latest features and improvements through regular releases for both client and management libraries.
*   **Easy to Get Started:** Get started with the latest packages from the Azure SDK via the easy to use packages.

## Getting Started

Azure SDK for Python libraries are supported on Python 3.9 or later. To get started with a specific library, refer to the `README.md` (or `README.rst`) file located in the library's project folder within the `/sdk` directory. For more information, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The Azure SDK for Python organizes libraries into the following categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

New releases of client libraries are announced as **GA** and several are in **preview**, providing access to existing resources. These libraries share core functionalities found in [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) such as retries, logging, transport protocols, and authentication. You can learn more about these libraries by reading guidelines that they follow [here](https://azure.github.io/azure-sdk/python/guidelines/index.html). You can find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Note:** For production, use stable, non-preview libraries.

### Client: Previous Versions

These are the last stable versions for production-ready Azure usage. While they may not include all features or adhere to all guidelines, they provide wider service coverage.

### Management: New Releases

The new set of management libraries follows the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They provide core capabilities that are shared across Azure SDKs, and allow for provision and management of resources. Documentation and code samples can be found [here](https://aka.ms/azsdk/python/mgmt). A migration guide from older versions can be found [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). You can find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

>   **Note:** Use stable, non-preview libraries for production. If you upgrade to the new SDK versions and experience authentication issues, refer to the migration guide.

### Management: Previous Versions

For a complete list of management libraries enabling you to provision and manage Azure resources, [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries can be identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`

## Need Help?

*   Comprehensive documentation is available at our [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   Search [previous questions](https://stackoverflow.com/questions/tagged/azure+python) or ask new ones on StackOverflow using `azure` and `python` tags.

## Data Collection

The software collects information about you and your use of the software and sends it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described below. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default. To opt-out, create a subclass of `UserAgentPolicy` with an empty `on_request` method. Then pass an instance of this class as `user_agent_policy` during client creation.

**Example:**
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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. Further information is available in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details. All contributions require a Contributor License Agreement (CLA), which you will be prompted to complete via the CLA-bot upon submitting a pull request.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or comments.