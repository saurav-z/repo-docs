# Azure SDK for Python: Build Robust Python Applications with Ease

**Empower your Python applications with seamless integration with Azure services using the official Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, databases, networking, and more.
*   **Modular Libraries:** Choose from individual, well-defined libraries for specific Azure services, promoting smaller package sizes and targeted dependencies.
*   **Client & Management Libraries:** Access both client libraries for interacting with existing resources and management libraries to provision and manage resources.
*   **Modern Design Guidelines:** Benefit from libraries that adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), providing a consistent and intuitive developer experience.
*   **Robust Authentication:** Leverage secure and reliable authentication methods, including Azure Active Directory (Azure AD) and Managed Identities.
*   **Simplified Development:** Streamline your development process with features like automatic retries, logging, and built-in support for transport protocols.
*   **Up-to-Date Documentation:** Access extensive documentation and code samples to help you get started quickly.
*   **Telemetry Control:** Control the data collection behavior via a disable telemetry option.
*   **Community Support:** Get help via documentation, GitHub issues, and Stack Overflow.

## Getting Started

### Prerequisites

The client libraries are supported on Python 3.9 or later. See [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.

### Available Packages

*   **Client Libraries (New Releases):** GA and preview libraries for interacting with existing resources. View the latest packages at [Azure SDK for Python - Releases](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
*   **Client Libraries (Previous Versions):** Stable versions for production use.
*   **Management Libraries (New Releases):** Libraries for provisioning and managing Azure resources, following Azure SDK design guidelines. Access the latest package info [Azure SDK for Python - Management Releases](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).
*   **Management Libraries (Previous Versions):**  A complete list of management libraries available [Azure SDK for Python - All Releases](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

### Installation

To install a specific service library, refer to the `README.md` (or `README.rst`) file located in the library's project folder, found within the `/sdk` directory of the repository.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **GitHub Issues:** [File an issue](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search or ask questions using the `azure` and `python` tags.

## Data Collection & Telemetry Configuration

The SDK collects data to improve products and services. Telemetry is enabled by default, but can be disabled during client construction. For details and configuration options, please see the "Telemetry Configuration" section in the original README or the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

## Reporting Security Issues

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

Contributions are welcome! See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details. This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).