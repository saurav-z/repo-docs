# Azure CLI: Manage Azure Resources from Your Command Line

**The Azure CLI is a powerful cross-platform command-line tool for managing Azure services, allowing you to control your cloud infrastructure with efficient and automated scripts.**  Learn more and contribute at the [original repo](https://github.com/Azure/azure-cli).

## Key Features of the Azure CLI

*   **Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Coverage:** Supports a wide range of Azure services and resources.
*   **Scripting and Automation:**  Ideal for automating tasks and integrating with your DevOps pipelines.
*   **Tab Completion:** Saves time and reduces errors with built-in tab completion for commands and parameters.
*   **Flexible Output:** Choose your preferred output format (JSON, table, tsv) or use the `--query` parameter with JMESPath for customized results.
*   **Exit Codes for Scripting:**  Provides standardized exit codes for easy error handling in your scripts.
*   **Integration with VS Code:** Utilize the Azure CLI Tools extension for IntelliSense, snippets, and integrated terminal support.

## Getting Started

*   **Installation:**  Refer to the [Install Guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed instructions.
*   **Troubleshooting:**  Consult the [Install Troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md) guide for common installation issues.
*   **Quick Start:** Begin with the ["Get Started" Guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   **Command Syntax:** `az [group] [subgroup] [command] {parameters}`
*   **Get Help:** Use the `-h` parameter to get help for commands and groups.  For example, `az storage -h` or `az vm create -h`.

## Example Usage

*   **List VMs with custom output:** `az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"`
*   **Configure Output Format:** Customize the output using `az configure`.

## Common Scenarios

*   **Output Formatting:** Control output format (JSON, table, or tsv) for readability.
*   **Passing Values:** Chain commands by passing values from one command to another.
*   **Async Operations:** Manage asynchronous operations effectively.
*   **Resource Management:** Utilize generic resource commands with `az resource`.
*   **REST API Access:**  Execute REST API calls using the `az rest` command.

For more in-depth information, refer to [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).

## Developer Installation & Edge Builds

*   **Docker:** Pre-configured Docker image available.  See [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list).
*   **Edge Builds:** Get the latest builds from the `dev` branch for testing new features.  Links for various platforms are provided in the original readme.  See the original readme for installation instructions.
*   **Builds of Arbitrary Commit/PR:**  [Try new features before release](doc/try_new_features_before_release.md)

## Contributing

*   **Report Issues:**  File bugs and issues in the [GitHub Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Provide Feedback:**  Use the `az feedback` command or contact the team at azpycli@microsoft.com.
*   **Contribute Code:**  Follow the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) to contribute.
*   **Code of Conduct:** This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

## Data Collection

The software collects information about your usage to improve the product. Telemetry collection is on by default. To opt out, run `az config set core.collect_telemetry=no`.  See the original readme for more details.