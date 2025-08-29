# Azure CLI: Command-Line Interface for Azure Cloud Management

**Manage your Azure cloud resources efficiently and effectively with the Azure CLI.**  ([Original Repository](https://github.com/Azure/azure-cli))

The Azure CLI is a cross-platform command-line interface (CLI) that allows you to manage Azure cloud resources from your terminal. It's a powerful tool for automation, scripting, and interactive use, available across Windows, macOS, and Linux.

**Key Features:**

*   **Cross-Platform Compatibility:**  Use the CLI on Windows, macOS, and Linux.
*   **Comprehensive Resource Management:** Manage a wide range of Azure services, including virtual machines, storage, networking, and more.
*   **Automation-Friendly:** Automate tasks and integrate with your scripts using a robust set of commands.
*   **Interactive Experience:** Explore and manage resources interactively from your terminal.
*   **Tab Completion:** Speed up your workflow with tab completion for commands and parameters.
*   **Querying with JMESPath:**  Customize output using the `--query` parameter and JMESPath syntax.
*   **Flexible Output Formats:** Display results in JSON, table, or TSV formats.
*   **Visual Studio Code Integration:** Enhance your workflow with the Azure CLI Tools extension, providing IntelliSense, snippets, and more.

## Installation

Refer to the official [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed installation instructions for your operating system.  Troubleshooting tips are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

```bash
$ az [group] [subgroup] [command] {parameters}
```

*   Get started by using the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   Use the `-h` parameter for help with any command:  `az storage -h` or `az vm create -h`.

## Highlights and Examples

*   **Tab Completion:**  `az vm show -g [tab][tab]`
*   **Query:**  `az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"`
*   **Exit Codes:** (See Original README)

## Effective Use and Common Scenarios

*   Output formatting (json, table, or tsv)
*   Pass values from one command to another
*   Async operations
*   Generic update arguments
*   Generic resource commands - `az resource`
*   REST API command - `az rest`
*   Quoting issues
*   Work behind a proxy
*   Concurrent builds

For more details, see the [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).

## Samples and Snippets

*   Explore the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) for usage examples.
*   See the [overview documentation](https://learn.microsoft.com/cli/azure/overview).

## Azure CLI Tools for Visual Studio Code

Enhance your workflow with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for Visual Studio Code.

## Data Collection and Telemetry

The Azure CLI collects data to improve its functionality.  You can opt-out by running `az config set core.collect_telemetry=no`.  Review the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) for more information.

## Reporting Issues and Providing Feedback

*   Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section.
*   Provide feedback from the command line using `az feedback`.

## Developer Installation (See Original README)

*   **Docker** (See Original README)
*   **Edge Builds** (See Original README)
*   **Get builds of arbitrary commit or PR** (See Original README)
*   **Developer setup** (See Original README)

## Contributing

Contribute to the Azure CLI project following the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).  This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).