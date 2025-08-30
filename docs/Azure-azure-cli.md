# Azure CLI: Command-Line Interface for Managing Azure Resources

**Manage your Azure cloud resources efficiently and securely with the Azure CLI, a powerful, cross-platform command-line tool.** ([Back to Original Repo](https://github.com/Azure/azure-cli))

**Key Features:**

*   **Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Coverage:** Manage a wide range of Azure services and resources.
*   **Scripting and Automation:** Ideal for automating tasks and integrating with scripts.
*   **Tab Completion:** Improves efficiency with tab completion for commands and parameters.
*   **Flexible Output:** Customize your output with JSON, table, and tsv formats.
*   **Querying:** Utilize JMESPath queries for filtering and formatting results.
*   **VS Code Integration:** Leverage the Azure CLI Tools extension for enhanced development, including IntelliSense, snippets, and in-editor command execution.

## Installation

*   Detailed installation instructions can be found in the [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli).
*   Troubleshooting common installation issues is available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

The general command structure is as follows:

```bash
az [group] [subgroup] [command] {parameters}
```

### Get Started

Refer to the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2) for detailed instructions.

To get help and usage information for a specific command, use the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

### Highlights

[Insert Azure CLI Highlight Reel GIF here (from original README)]

**Key Functionality:**

*   **Tab Completion:** Improve your efficiency with tab completion for groups, commands, and parameters.
    ```bash
    # Example:
    $ az vm show -g [tab][tab]  # Lists resource groups
    ```
*   **Query:** Use the `--query` parameter and JMESPath to customize your output.
    ```bash
    $ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
    ```
*   **Exit Codes:** Use exit codes for scripting.

    | Exit Code | Scenario                                                    |
    | --------- | ----------------------------------------------------------- |
    | 0         | Command ran successfully.                                  |
    | 1         | Generic error; server returned bad status code, CLI validation failed, etc. |
    | 2         | Parser error; check input to command line.                 |
    | 3         | Missing ARM resource; used for existence check from `show` commands. |

### Common scenarios and use Azure CLI effectively

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

### More samples and snippets

*   Explore more usage examples on the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) or at [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

### Write and run commands in Visual Studio Code

*   Install the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for VS Code.
    *   Get IntelliSense.
    *   Use code snippets.
    *   Run commands in the integrated terminal.
    *   View output in a side-by-side editor.
    *   See documentation on hover.
    *   Display current subscription and defaults in status bar.
    *   See [microsoft/vscode-azurecli#48](https://github.com/microsoft/vscode-azurecli/issues/48) for enabling IntelliSense for other file types.

[Insert Azure CLI Tools in Action GIF here (from original README)]

## Data Collection

*   The Azure CLI collects usage data to improve the tool.
*   You can disable telemetry using `az config set core.collect_telemetry=no`.
*   For more information, see the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).

## Reporting issues and feedback

*   File issues in the [Issues](https://github.com/Azure/azure-cli/issues) section.
*   Provide feedback from the command line using the `az feedback` command.

## Developer installation

### Docker

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge builds

*   Get the latest builds from the `dev` branch.

|      Package      | Link                                       |
|:-----------------:|:-------------------------------------------|
|        MSI        | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
|      RPM el8      | https://aka.ms/InstallAzureCliRpmEl8Edge   |

### Edge build installation commands
*   See the original README for detailed instructions on installation.

### Get builds of arbitrary commit or PR

*   See [Try new features before release](doc/try_new_features_before_release.md)

## Developer setup

*   See [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   See [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   See [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute code

*   This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
*   Contribute following the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).