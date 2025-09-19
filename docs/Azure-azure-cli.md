# Azure CLI: Your Command-Line Interface for Azure Cloud Management

**Manage and automate your Azure cloud resources with ease using the powerful and versatile Azure CLI.** ([Back to Original Repo](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform:** Available on Windows, macOS, and Linux.
*   **Comprehensive Azure Coverage:** Supports a vast array of Azure services and resources.
*   **Easy-to-Use Syntax:**  Offers a consistent and intuitive command structure: `az [group] [subgroup] [command] {parameters}`.
*   **Interactive Cloud Shell:** Run commands directly from your browser via [Azure Cloud Shell](https://portal.azure.com/#cloudshell).
*   **Tab Completion:**  Enhances productivity with tab completion for commands and parameters.
*   **Flexible Output Options:** Customize output with `--query` and [JMESPath](http://jmespath.org/) for tailored results.
*   **Scripting Support:** Provides exit codes for seamless integration into scripts and automation workflows.
*   **VS Code Integration:** Enhance your development experience with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension.

## Installation

Get started quickly by following the detailed instructions in the [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli).

Troubleshooting tips are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

*   **Getting Started:** Explore the Azure CLI with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   **Help and Documentation:** Get help for commands with the `-h` parameter (e.g., `az storage -h`).

## Examples and Key Concepts

*   **Tab Completion:** Simplify command entry with tab completion for groups, commands, and parameters (e.g., `az vm show -g [tab][tab]`).
*   **Querying:** Customize output using the `--query` parameter and JMESPath for precise information retrieval.
*   **Exit Codes:** Understand the exit codes for scripting and automation.
*   **Output Formatting:** Choose your preferred output format (json, table, or tsv) or configure a default using the `az configure` command.

### Scenarios and Tips

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## Additional Resources

*   **Samples:** Find more usage examples in our [GitHub samples repo](http://github.com/Azure/azure-cli-samples).
*   **Documentation:** Explore the comprehensive documentation on [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

## Developer Installation

### Docker

Run Azure CLI inside a Docker container:

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Access the latest builds from the `dev` branch.

**Available Versions:**
*   MSI: https://aka.ms/InstallAzureCliWindowsEdge
*   Homebrew Formula: https://aka.ms/InstallAzureCliHomebrewEdge
*   Ubuntu Bionic Deb: https://aka.ms/InstallAzureCliBionicEdge
*   Ubuntu Focal Deb: https://aka.ms/InstallAzureCliFocalEdge
*   Ubuntu Jammy Deb: https://aka.ms/InstallAzureCliJammyEdge
*   RPM el8: https://aka.ms/InstallAzureCliRpmEl8Edge

Example: Install the latest Homebrew edge build:

```bash
curl --location --silent --output azure-cli.rb https://aka.ms/InstallAzureCliHomebrewEdge
brew install --build-from-source azure-cli.rb
```

### Development Setup

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

*   [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
*   [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate)

## Reporting Issues and Feedback

*   File an issue in the [Issues](https://github.com/Azure/azure-cli/issues) section of our GitHub repo.
*   Provide feedback from the command line using the `az feedback` command.