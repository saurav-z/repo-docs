# Azure CLI: Manage Your Azure Resources with Ease

**The Azure CLI is a powerful, cross-platform command-line interface that empowers you to manage and configure your Azure resources with efficiency and speed.**  [Get Started with the Azure CLI](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Azure Coverage:** Supports a vast array of Azure services and features.
*   **Intuitive Command Structure:**  Uses a clear and consistent `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:**  Enhances productivity with tab completion for commands, groups, and parameters.
*   **Powerful Querying with JMESPath:**  Customize output using the `--query` parameter and JMESPath syntax.
*   **Scripting Friendly Exit Codes:** Provides exit codes for scripting integration.
*   **VS Code Integration:**  The Azure CLI Tools extension offers features like IntelliSense, snippets, and command execution within Visual Studio Code.

## Getting Started

### Installation

Detailed installation instructions are available in the [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli).  Troubleshooting tips can be found in the [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md) document.

### Basic Usage

Use the following format to run commands:

```bash
$ az [group] [subgroup] [command] {parameters}
```

Start with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2) for an in-depth introduction. Get help with the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

## Highlights & Core Concepts

The Azure CLI offers features designed to make your experience smoother and more efficient:

![Azure CLI Highlight Reel](doc/assets/AzBlogAnimation4.gif)

*   **Tab Completion:**  Speed up command entry.

    ```bash
    # looking up resource group and name
    $ az vm show -g [tab][tab]
    AccountingGroup   RGOne  WebPropertiesRG

    $ az vm show -g WebPropertiesRG -n [tab][tab]
    StoreVM  Bizlogic

    $ az vm show -g WebPropertiesRG -n Bizlogic
    ```
*   **Querying:**  Customize your output with `--query` and JMESPath.

    ```bash
    $ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
    Name                    Os
    ----------------------  -------
    storevm                 Linux
    bizlogic                Linux
    demo32111vm             Windows
    dcos-master-39DB807E-0  Linux
    ```
*   **Exit Codes:**  Integrate commands in scripts with confidence.

    | Exit Code | Scenario                                     |
    | --------- | -------------------------------------------- |
    | 0         | Command ran successfully.                    |
    | 1         | Generic error; bad status code, validation failed. |
    | 2         | Parser error; check input.                  |
    | 3         | Missing ARM resource; for `show` commands.   |

## Common Scenarios & Effective Use

Optimize your use of the Azure CLI with these helpful tips:  Refer to the [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively) for in-depth instructions.

*   Output formatting (json, table, or tsv)
*   Pass values from one command to another
*   Async operations
*   Generic update arguments
*   Generic resource commands - `az resource`
*   REST API command - `az rest`
*   Quoting issues
*   Work behind a proxy
*   Concurrent builds

## Resources for Further Learning

*   **Samples and Snippets:** Explore more examples at the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

## Azure CLI Tools for Visual Studio Code

Enhance your Azure CLI workflow in Visual Studio Code with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension:

*   IntelliSense for commands and arguments.
*   Command snippets for automatic argument insertion.
*   Run and output commands in the integrated terminal and side-by-side editor.
*   Inline documentation on hover.
*   Subscription and default display in the status bar.

![Azure CLI Tools in Action](https://github.com/microsoft/vscode-azurecli/blob/main/images/in_action.gif?raw=true)

## Data Collection and Telemetry

The Azure CLI collects usage data to help improve the service. You can opt-out by running `az config set core.collect_telemetry=no`.  See the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkID=824704) for details.

## Reporting Issues and Feedback

*   **File Issues:** Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section of the GitHub repository.
*   **Provide Feedback:** Use the `az feedback` command from the command line.
*   **Contact the Team:** \[Microsoft internal] Email the developer team at azpycli@microsoft.com.

## Developer Installation

### Docker
```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Get the latest builds from the `dev` branch. Find the latest versions here: [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list).

|      Package      | Link                                       |
|:-----------------:|:-------------------------------------------|
|        MSI        | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
|      RPM el8      | https://aka.ms/InstallAzureCliRpmEl8Edge   |

### Get builds of arbitrary commit or PR

[Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

If you would like to setup a development environment and contribute to the CLI, see:

[Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)

[Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)

[Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contributing

Contributions are welcome!  Please adhere to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).