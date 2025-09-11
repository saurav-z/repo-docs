# Azure CLI: Manage Azure Resources with the Command Line

**Unlock the power of Azure with the Azure CLI, a cross-platform command-line tool for managing and interacting with your Azure resources.** ([Original Repository](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform Compatibility:** Works on Windows, macOS, and Linux.
*   **Resource Management:** Easily create, manage, and delete Azure resources like virtual machines, storage accounts, and more.
*   **Command-Line Efficiency:** Automate tasks and script complex workflows with concise commands.
*   **Interactive Experience:** Explore resources and options with tab completion and helpful documentation.
*   **Flexible Output:** Customize results with JSON, table, or tsv formats, and query results using JMESPath.
*   **Extensible Functionality:** Integrate with Visual Studio Code for enhanced command authoring and execution.
*   **Telemetry and Configuration:** Customize the level of telemetry.

## Installation

Detailed installation instructions are available in the [Azure CLI install guide](https://learn.microsoft.com/cli/azure/install-azure-cli).

## Usage

Use the `az` command followed by resource group, subgroup, and command: `az [ group ] [ subgroup ] [ command ] {parameters}`

**Example:**

```bash
$ az vm create -g MyResourceGroup -n MyVM --image UbuntuLTS
```

For help, use the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

## Highlights & Benefits

*   **Tab Completion:** Save time and avoid errors with tab completion for commands and parameters.
*   **Querying Output:** Use the `--query` parameter and JMESPath to filter and format command output, extracting specific data.
*   **Exit Codes:** Script effectively with standard exit codes for success (0) and various error scenarios (1, 2, 3).

## Common Scenarios and Effective Use

Find tips for efficient usage in the [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).

*   Output formatting (json, table, or tsv)
*   Pass values from one command to another
*   Async operations
*   Generic update arguments
*   Generic resource commands - `az resource`
*   REST API command - `az rest`
*   Quoting issues
*   Work behind a proxy
*   Concurrent builds

## More Examples

Explore more usage examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) or the [Azure CLI overview](https://learn.microsoft.com/cli/azure/overview).

## Visual Studio Code Integration

Enhance your CLI experience with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for Visual Studio Code.
*   IntelliSense for Commands
*   Snippets for Commands
*   Run Commands in the Terminal
*   Show Documentation on Hover

## Telemetry

The Azure CLI collects usage data to improve the tool. You can opt-out by running: `az config set core.collect_telemetry=no`

## Reporting Issues and Providing Feedback

Report bugs and provide feedback via:

*   [GitHub Issues](https://github.com/Azure/azure-cli/issues)
*   `az feedback` command in the CLI

## Developer Information

### Developer Installation

*   [Docker](#docker)
*   [Edge Builds](#edge-builds)
*   [Developer Setup](#developer-setup)

### Contribute Code

Learn more about contributing by following the instructions provided in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).