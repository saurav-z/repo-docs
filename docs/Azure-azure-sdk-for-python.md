# Azure SDK for Python: Simplify Cloud Development

The Azure SDK for Python provides a comprehensive set of libraries that empower developers to easily build, manage, and deploy applications on the Microsoft Azure cloud platform. **Get started with the official Azure SDK for Python to build robust and scalable cloud applications.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

## Key Features

*   **Broad Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Modernized Libraries:** Utilize new and improved client and management libraries adhering to the Azure SDK Design Guidelines for Python.
*   **Simplified Authentication:** Leverage the Azure Identity library for secure and streamlined authentication.
*   **Comprehensive Documentation:** Benefit from extensive documentation, code samples, and migration guides to accelerate your development.
*   **Cross-Platform Compatibility:** Develop on Python 3.9 or later.
*   **Telemetry and Data Collection:** Control data collection to ensure your privacy preferences are met.
*   **Easy Installation:** Install packages using pip or other package managers.

## Getting Started

The Azure SDK for Python is structured with individual libraries for each Azure service, allowing for focused development. To begin:

1.  **Explore Available Packages:** Browse the `/sdk` directory within the repository to find libraries for specific services.
2.  **Review Service-Specific `README.md`:** Each service library includes its own `README.md` (or `README.rst`) with detailed instructions.
3.  **Check Prerequisites:** Ensure you have Python 3.9 or later installed. Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more information.

## Package Categories

*   **Client Libraries (New Releases):**  GA (Generally Available) and preview libraries for consuming and interacting with existing resources (e.g., uploading blobs).
*   **Client Libraries (Previous Versions):** Stable versions of libraries providing functionalities similar to the preview ones, with wider coverage.
*   **Management Libraries (New Releases):**  Management libraries aligned with Azure SDK Design Guidelines for Python, offering core capabilities like authentication, HTTP pipelines, and error handling.
*   **Management Libraries (Previous Versions):**  Management libraries for provisioning and managing Azure resources.

## Resources

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **GitHub Issues:** [Report issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search or ask questions using `azure` and `python` tags.
*   **Most Up-to-Date Package Lists:**
    *   [Client Libraries (New Releases)](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
    *   [Management Libraries (New Releases)](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)
    *   [Management Libraries (All)](https://azure.github.io/azure-sdk/releases/latest/all/python.html)
*   **Migration Guide:** Transition from older versions of libraries -  [Migration Guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

## Data Collection and Telemetry

The Azure SDK for Python collects data about your usage to improve services.

*   **Telemetry Configuration:** Disable telemetry during client construction using `NoUserAgentPolicy`
    *   See original README for a code example of how to do so.

## Reporting Security Issues

Report security vulnerabilities privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

Contribute to the Azure SDK for Python by following the guidelines in the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). All contributors must agree to a Contributor License Agreement (CLA).

*   **Code of Conduct:** This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).