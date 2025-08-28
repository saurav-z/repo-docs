# Azure CLI: Manage Your Azure Resources From the Command Line

**The Azure CLI is a powerful, cross-platform command-line interface (CLI) for managing and interacting with your Azure resources, providing a streamlined experience for cloud management.** ([Original Repo](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform:** Available on Windows, macOS, and Linux.
*   **Comprehensive:** Manage a vast array of Azure services.
*   **Automation-Friendly:**  Ideal for scripting and automating tasks.
*   **Tab Completion:** Improves efficiency with command suggestions.
*   **Flexible Output:** Customize output with `--query` and JMESPath for specific data retrieval.
*   **Exit Codes:** Consistent exit codes for scripting and automation.
*   **VS Code Integration:** Enhance your workflow with the Azure CLI Tools extension.

## Installation

Follow the comprehensive [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed instructions on your operating system.  Troubleshooting tips can be found at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

The general command structure is:

```bash
az [group] [subgroup] [command] {parameters}
```

Get started quickly with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).  For help with any command, use the `-h` parameter:

```bash
az storage -h
az vm create -h
```

## Highlights & Examples

![Azure CLI Highlight Reel](doc/assets/AzBlogAnimation4.gif)

### Tab Completion
```bash
$ az vm show -g [tab][tab]
AccountingGroup   RGOne  WebPropertiesRG

$ az vm show -g WebPropertiesRG -n [tab][tab]
StoreVM  Bizlogic

$ az vm show -g WebPropertiesRG -n Bizlogic
```

### Query with JMESPath
```bash
$ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
Name                    Os
----------------------  -------
storevm                 Linux
bizlogic                Linux
demo32111vm             Windows
dcos-master-39DB807E-0  Linux
```

### Exit Codes

| Exit Code | Scenario                                                              |
| --------- | --------------------------------------------------------------------- |
| 0         | Command ran successfully.                                             |
| 1         | Generic error; server returned bad status code, CLI validation failed, etc. |
| 2         | Parser error; check input to command line.                             |
| 3         | Missing ARM resource; used for existence check from `show` commands.   |

## Effective Use Cases & Resources

Learn to optimize your Azure CLI experience with these key resources:

*   **Output formatting (json, table, or tsv):** [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   **Pass values from one command to another:** [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   **Async operations:** [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   **Generic update arguments:** [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   **Generic resource commands - `az resource`:** [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   **REST API command - `az rest`:** [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   **Quoting issues:** [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   **Work behind a proxy:** [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   **Concurrent builds:** [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

Explore more examples in our [GitHub samples repo](http://github.com/Azure/azure-cli-samples) or [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

## Azure CLI Tools for Visual Studio Code

Enhance your development with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension in Visual Studio Code. Key features include:

*   IntelliSense for commands and arguments
*   Command snippets
*   Integrated terminal execution
*   Side-by-side editor output
*   Documentation on hover
*   Subscription & defaults display

![Azure CLI Tools in Action](https://github.com/microsoft/vscode-azurecli/blob/main/images/in_action.gif?raw=true)

## Data Collection and Telemetry

The Azure CLI collects usage data to improve the service. You can opt-out by running `az config set core.collect_telemetry=no`. View the Microsoft privacy statement at [https://go.microsoft.com/fwlink/?LinkID=824704](https://go.microsoft.com/fwlink/?LinkID=824704).

## Reporting Issues and Feedback

Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section of our GitHub repo. Provide feedback directly from the command line using `az feedback`.

## Developer Installation

### Docker

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Install the latest builds from the `dev` branch:

*   **MSI:** [https://aka.ms/InstallAzureCliWindowsEdge](https://aka.ms/InstallAzureCliWindowsEdge)
*   **Homebrew:** `curl --location --silent --output azure-cli.rb https://aka.ms/InstallAzureCliHomebrewEdge && brew install --build-from-source azure-cli.rb`
*   **Ubuntu Jammy:** `curl --location --silent --output azure-cli_jammy.deb https://aka.ms/InstallAzureCliJammyEdge && dpkg -i azure-cli_jammy.deb`
*   **RPM el8:** `dnf install -y $(curl --location --silent --output /dev/null --write-out %{url_effective} https://aka.ms/InstallAzureCliRpmEl8Edge)`

*   **Pip3:**
```bash
$ python3 -m venv env
$ . env/bin/activate
$ pip3 install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --upgrade-strategy=eager
```

### Get builds of arbitrary commit or PR
[Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

To contribute to the CLI, follow the instructions in:

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute Code

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For details, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).  Follow the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) for contributions.