# Azure SDK for Python: Build Powerful Python Applications on Azure

**Get started with the Azure SDK for Python and seamlessly integrate your Python applications with Microsoft Azure services.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Simplified Development:** Utilize well-documented, user-friendly Python libraries that align with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).
*   **Latest Releases & Production-Ready Libraries:** Explore both new and previous versions of client and management libraries, with clear distinctions for GA (Generally Available) and preview releases, ensuring stability and access to the latest features.
*   **Consistent Design:** Benefit from core shared functionalities like retries, logging, transport protocols, and authentication, found in the `azure-core` library.
*   **Flexible Authentication:** Integrate easily with Azure Active Directory, managed identities, and other authentication methods.
*   **Management Libraries:** Provision and manage Azure resources, including the intuitive Azure Identity library, an HTTP Pipeline with custom policies, error-handling, distributed tracing, and much more.

## Getting Started

*   **Choose your service:** Select the specific Azure service you need (e.g., Blob Storage, Cosmos DB).
*   **Install the library:** Install the corresponding Python package for that service using `pip`.  Service-specific `README.md` files in the `/sdk` directory provide detailed instructions.
*   **Follow the documentation:** Consult the official [Azure SDK for Python documentation](https://aka.ms/python-docs) for in-depth guides, code samples, and API references.

## Available Packages

Libraries are categorized for client and management use, with distinctions between new and previous releases.  Find the most up-to-date package lists:

*   **Client - New Releases:** [Latest packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
*   **Client - Previous Versions:** Stable, production-ready versions.
*   **Management - New Releases:** [Latest packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)
*   **Management - Previous Versions:** Provision and manage Azure resources. Complete list [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries are identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`

## Prerequisites

*   Python 3.9 or later is required.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** File an issue via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Search or ask questions on StackOverflow using `azure` and `python` tags.

## Data Collection & Telemetry

The Azure SDK for Python collects telemetry data to improve the service. Users can disable telemetry. Instructions are in the original README, duplicated below.

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

## Security

*   Report security issues and bugs privately to the Microsoft Security Response Center (MSRC):  <secure@microsoft.com>.

## Contributing

*   See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.
*   This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).