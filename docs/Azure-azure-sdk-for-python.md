# Azure SDK for Python: Build Robust Applications on Microsoft Azure

**Empower your Python applications with seamless integration and powerful tools by leveraging the official Azure SDK for Python.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, databases, networking, and more.
*   **Modern and Consistent Design:** Leverage SDKs designed with a consistent API surface, offering improved usability and maintainability, as well as adherence to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).
*   **Latest Releases:** Get access to the newest packages and features for both client and management libraries.
*   **Production-Ready Libraries:** Utilize stable, non-preview libraries for reliable performance in your production environments.
*   **Management and Client Libraries:** Utilize client and management libraries categorized into new and previous versions.
*   **Management Libraries:** Provision and manage Azure resources.
*   **Active Development:** Benefit from ongoing development and updates, ensuring compatibility with the latest Azure services and features.
*   **Easy to Get Started:** Each service has a separate set of libraries and is available to use in the `/sdk` directory.
*   **Detailed Documentation:** Get your hands on the documentation for the Azure SDK for Python at [Azure SDK for Python documentation](https://aka.ms/python-docs).

## Getting Started

To begin, explore the libraries available in the `/sdk` directory. Each service has its own set of libraries for convenient usage.

### Prerequisites

*   Python 3.9 or later.
*   Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more information.

## Packages Available

*   **Client Libraries:** New Releases, Previous Versions
*   **Management Libraries:** New Releases, Previous Versions

### Client: New Releases

These libraries are GA and preview and help with the usage and consumption of Azure resources.  They share core functionalities like retries, logging, and authentication protocols. The [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library has several core functionalities such as retries, logging, transport protocols, authentication protocols, etc. Read the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html). Get the latest packages from [our page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> NOTE: Use non-preview libraries for production.

### Client: Previous Versions

These provide similar functionality to the preview releases.

### Management: New Releases

These management libraries follow [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and provide core capabilities shared across all Azure SDKs. Refer to the [documentation](https://aka.ms/azsdk/python/mgmt) and the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for more info. Find all of the new packages on [our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> NOTE: Ensure you're using a stable, non-preview library for production and follow the migration guide if upgrading packages.

### Management: Previous Versions

For a complete list of management libraries, [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Identify libraries by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **GitHub Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **StackOverflow:** Search with `azure` and `python` tags.

## Data Collection and Telemetry

The SDK collects data to improve products and services.

*   **Telemetry Policy:** [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy)
*   **Opt-out:** Disable telemetry with `NoUserAgentPolicy`.
*   See the example code in the original `README.md` for more information.

## Reporting Security Issues

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

Contributions are welcome!

*   **Contributing Guide:** [Contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md)
*   **CLA:** Contributions require a Contributor License Agreement.
*   **Code of Conduct:** [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)