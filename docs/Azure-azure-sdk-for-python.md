# Azure SDK for Python: Simplify Cloud Development with Powerful Python Libraries

Develop robust and scalable applications on Azure with the official **Azure SDK for Python** ([Original Repo](https://github.com/Azure/azure-sdk-for-python)).

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modern Design:** The SDK adheres to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), providing a consistent and intuitive development experience.
*   **Client and Management Libraries:** Utilize client libraries for interacting with existing resources (e.g., uploading a blob) and management libraries for provisioning and managing Azure resources.
*   **Latest Releases:** Benefit from new features, performance improvements, and bug fixes with the latest [Client - New Releases](https://azure.github.io/azure-sdk/releases/latest/index.html#python) and [Management - New Releases](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).
*   **Stable Previous Versions:** Use stable, production-ready [Client - Previous Versions](#client-previous-versions) and [Management - Previous Versions](#management-previous-versions) for wider service coverage.
*   **Azure Identity Integration:** Leverage the intuitive Azure Identity library for secure authentication.
*   **Robust Core Functionality:** Built-in features like retries, logging, transport protocols, and authentication protocols are available in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Detailed Documentation:** Comprehensive documentation is available at [Azure SDK for Python documentation](https://aka.ms/python-docs) to help you get started.

## Getting Started

Each Azure service has a dedicated set of Python libraries.  Find the specific library you need within the `/sdk` directory and explore the corresponding `README.md` (or `README.rst`) file for detailed instructions.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The Azure SDK for Python offers libraries categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Explore [Stack Overflow](https://stackoverflow.com/questions/tagged/azure+python) with `azure` and `python` tags.

## Data Collection and Telemetry Configuration

This software collects usage data, which can be disabled by using the `NoUserAgentPolicy` class and passing it as a `user_agent_policy` parameter when constructing client objects. See the original README for detailed configuration steps.

## Security

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) via <secure@microsoft.com>.

## Contributing

Contribute to the project by following the instructions in the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).