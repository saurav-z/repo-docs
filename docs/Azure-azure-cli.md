# Azure CLI: Manage Your Azure Resources with Ease

**The Azure CLI is a cross-platform command-line interface (CLI) that provides a powerful and intuitive way to interact with your Azure cloud resources.**  [Learn more about the Azure CLI](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Azure Coverage:** Manage a wide range of Azure services, from virtual machines to storage and networking.
*   **Intuitive Command Structure:**  Uses a consistent `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:**  Enhances productivity with tab completion for commands, groups, and parameters.
*   **Powerful Querying with JMESPath:**  Customize your output with the `--query` parameter using the [JMESPath](http://jmespath.org/) query language.
*   **Scripting-Friendly Exit Codes:** Provides specific exit codes for various scenarios, making it ideal for automation.
*   **Integration with VS Code:**  Utilize the Azure CLI Tools extension for VS Code for features like IntelliSense, snippets, and integrated terminal execution.

## Getting Started

*   **Installation:** Follow the detailed [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) to get started. Troubleshooting tips are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).
*   **Azure Cloud Shell:**  Test the CLI directly from your browser with [Azure Cloud Shell](https://portal.azure.com/#cloudshell).
*   **Usage:** Explore commands with `az [group] -h` (e.g., `az storage -h`, `az vm create -h`).
*   **Get Started Guide:**  Refer to the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2) for in-depth instructions.

##  Common Scenarios and Effective Usage

Explore these resources to maximize your efficiency:

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## Examples and Snippets

*   **Tab Completion:**

    ```bash
    # looking up resource group and name
    $ az vm show -g [tab][tab]
    AccountingGroup   RGOne  WebPropertiesRG

    $ az vm show -g WebPropertiesRG -n [tab][tab]
    StoreVM  Bizlogic

    $ az vm show -g WebPropertiesRG -n Bizlogic
    ```

*   **Query Output:**

    ```bash
    $ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
    Name                    Os
    ----------------------  -------
    storevm                 Linux
    bizlogic                Linux
    demo32111vm             Windows
    dcos-master-39DB807E-0  Linux
    ```

##  Developer Installation and Setup

###  Docker

Preconfigured Docker image for Azure CLI: See available versions [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list).

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Get the latest builds from the `dev` branch.

|      Package      | Link                                       |
|:-----------------:|:-------------------------------------------|
|        MSI        | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
|      RPM el8      | https://aka.ms/InstallAzureCliRpmEl8Edge   |

###  Developer Setup Links

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

##  Reporting Issues and Contributing

*   **Report bugs:** File an issue in the [Issues](https://github.com/Azure/azure-cli/issues) section of the GitHub repo.
*   **Provide feedback:** Use the `az feedback` command.
*   **Contribute:** Follow the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).
*   **Code of Conduct:**  See the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).