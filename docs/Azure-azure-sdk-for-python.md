# Azure SDK for Python: Build Powerful Python Applications with Azure

**Develop robust and scalable Python applications that seamlessly integrate with Microsoft Azure using the comprehensive Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide array of Azure services, including storage, compute, networking, databases, and more.
*   **Client and Management Libraries:** Utilize both client libraries for interacting with existing resources and management libraries for provisioning and managing Azure resources.
*   **Modern Design:** Follows the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for consistency and ease of use.
*   **Authentication and Authorization:** Integrated with Azure Active Directory and other authentication methods for secure access.
*   **Asynchronous Operations:** Leverage asynchronous programming for improved performance and scalability.
*   **Telemetry Configuration:** Configure telemetry to disable data collection to suit your needs.
*   **Regular Updates:** Benefit from frequent updates and new features as Azure services evolve.
*   **Detailed Documentation:** Access comprehensive documentation to guide you through the SDK.

## Getting Started

Choose from a variety of libraries for specific Azure services, each with its own README.md for detailed usage information. Explore the `/sdk` directory to find service-specific libraries.

### Prerequisites

Ensure your environment meets these requirements:

*   Python 3.9 or later.
*   For detailed support policy, please read [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

Explore the available packages based on your needs:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These **GA** (General Availability) and **preview** packages enable you to interact with existing resources, such as uploading a blob. These libraries share core functionalities like retries, logging, and authentication protocols found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. Explore the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) to learn more.

Find the [most up-to-date list of new packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> NOTE: For production, use stable, non-preview libraries.

### Client: Previous Versions

These are the last stable versions of packages that provide production-ready Azure usage. Similar to the Preview ones, these libraries allow you to use and consume existing resources and interact with them, for example: upload a blob. They might not implement the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) or have the same feature set as the November releases. They do however offer wider coverage of services.

### Management: New Releases

The new management libraries adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They offer essential capabilities shared across all Azure SDKs, including the Azure Identity library, HTTP Pipeline, error handling, and distributed tracing.

Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> NOTE: For production use, choose stable, non-preview libraries. Refer to the migration guide if upgrading management libraries and experiencing authentication issues.

### Management: Previous Versions

For a comprehensive list of management libraries, check [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries may not have all the features of the new releases, but they offer extensive service coverage. Management libraries start with the `azure-mgmt-` namespace (e.g., `azure-mgmt-compute`).

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow ([azure](https://stackoverflow.com/questions/tagged/azure) and [python](https://stackoverflow.com/questions/tagged/python) tags)

## Data Collection

The SDK may collect data, which can be disabled. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry collection is on by default.
To opt out, disable telemetry at client construction, by defining a `NoUserAgentPolicy` class and passing it as `user_agent_policy=NoUserAgentPolicy()`

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

## Reporting security issues and security bugs

Report security issues privately to the [Microsoft Security Response Center (MSRC)](mailto:secure@microsoft.com).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.