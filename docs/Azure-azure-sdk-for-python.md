# Azure SDK for Python: Build Powerful Python Applications with Azure Services

**Unlock the full potential of Microsoft Azure with the official Azure SDK for Python, designed to make integrating Azure services into your Python applications seamless and efficient.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, databases, AI, and more.
*   **Simplified Development:**  Libraries tailored for each Azure service, enabling easy integration with your Python code.
*   **Modern Design:** Utilize core functionalities like retries, logging, and authentication protocols for consistent performance.
*   **Production Ready:** Choose from stable, non-preview libraries to ensure your code is ready for production use.
*   **Management Libraries:** Provision and manage Azure resources with intuitive libraries following the Azure SDK Design Guidelines.
*   **Version Support Policy:** Supports Python 3.9 or later.

## Getting Started

The Azure SDK for Python provides a modular approach, with individual libraries for each Azure service.  To begin, explore the `README.md` (or `README.rst`) file within each service library's project folder in the `/sdk` directory.  For detailed documentation, please visit our [Azure SDK for Python documentation](https://aka.ms/python-docs).

## Packages Available

Choose from a variety of packages to suit your needs:

*   **Client: New Releases:** GA and preview libraries for consuming and interacting with existing resources. See the [latest package list](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
*   **Client: Previous Versions:**  Stable, production-ready libraries for wider service coverage, offering similar functionalities as the new releases.
*   **Management: New Releases:**  New management libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), offering shared capabilities and an improved user experience. See the [latest package list](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html). Refer to the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for upgrading.
*   **Management: Previous Versions:**  Libraries for provisioning and managing Azure resources.  See the [complete list](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:**  Search and ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection

The SDK collects usage data to improve products.  You can opt-out of telemetry.

*   **Telemetry Configuration:**  Disable telemetry by creating a custom `NoUserAgentPolicy` and passing it during client construction.  See example code in the original README.
*   **Data Collection Details:** Learn more in the help documentation and Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

## Reporting Security Issues

Report security issues privately to the Microsoft Security Response Center (MSRC): <secure@microsoft.com>.

## Contributing

Contributions are welcome! See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) and adhere to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).