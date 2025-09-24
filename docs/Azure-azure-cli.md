# Azure CLI: Manage Azure Resources from Your Command Line

**The Azure CLI is a powerful, cross-platform command-line tool that enables you to manage your Azure resources efficiently.** ([Back to Original Repo](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Resource Management:**  Control a wide range of Azure services, from virtual machines to storage accounts and everything in between.
*   **Interactive and Scriptable:** Use commands interactively or integrate them into your scripts for automation.
*   **Tab Completion:**  Save time and reduce errors with built-in tab completion for commands and parameters.
*   **Flexible Output:** Format results as JSON, tables, or TSV, and query results with JMESPath.
*   **VS Code Integration:**  Enhance your coding experience with the Azure CLI Tools extension for Visual Studio Code, offering IntelliSense, snippets, and more.
*   **Edge Builds:** Stay ahead of the curve with access to the latest features and updates through "edge" builds.

## Installation

Easily install the Azure CLI on your preferred platform.

*   Refer to the detailed [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for comprehensive instructions.
*   Troubleshooting tips are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Getting Started

Quickly get up and running with the Azure CLI.

*   Explore the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2) for in-depth instructions.
*   Use the `-h` parameter for help with any command:

    ```bash
    az storage -h
    az vm create -h
    ```

## Core Usage

The basic structure of an Azure CLI command is:

```bash
$ az [ group ] [ subgroup ] [ command ] {parameters}
```

## Highlights & Examples

### Tab Completion

*   Enhance your workflow with tab-completion for groups, commands, and parameters.

    ```bash
    # example
    $ az vm show -g [tab][tab]
    ```

### Querying Output

*   Customize your output with the `--query` parameter and [JMESPath](http://jmespath.org/) syntax.

    ```bash
    # example
    $ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
    ```

### Exit Codes

*   Understand exit codes for scripting:

    *   `0`: Success
    *   `1`: Generic error
    *   `2`: Parser error
    *   `3`: Missing ARM resource

## Effective Usage

Learn how to use Azure CLI effectively.

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## Explore Further

*   Browse our [GitHub samples repo](http://github.com/Azure/azure-cli-samples) for more usage examples.
*   Visit [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview) for comprehensive documentation.

## Visual Studio Code Integration

Enhance your command-line workflow with the Azure CLI Tools extension for Visual Studio Code:

*   IntelliSense for commands and arguments.
*   Command snippets.
*   Run commands directly from the editor.
*   Output display in a side-by-side editor.
*   Documentation on hover.
*   Display subscription and defaults in status bar.
*   ![Azure CLI Tools in Action](https://github.com/microsoft/vscode-azurecli/blob/main/images/in_action.gif?raw=true)

## Data Collection & Telemetry

*   The Azure CLI collects usage data to improve the service.  You can opt-out by running `az config set core.collect_telemetry=no`.
*   See our privacy statement for more details: [https://go.microsoft.com/fwlink/?LinkID=824704](https://go.microsoft.com/fwlink/?LinkID=824704)

## Reporting Issues & Feedback

*   Report bugs via the [Issues](https://github.com/Azure/azure-cli/issues) section of our GitHub repo.
*   Provide feedback from the command line with the `az feedback` command.

## Developer Installation & Contribution

*   See developer setup:

    *   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
    *   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
    *   [Code Generation](https://github.com/Azure/aaz-dev-tools)
*   Contribute code by following the instructions provided in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).
*   This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).