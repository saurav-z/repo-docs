# Azure CLI: Manage Your Azure Resources with Ease

**The Azure CLI is a powerful, cross-platform command-line tool for managing Azure services, offering a streamlined experience for developers and IT professionals.**  [Learn more about Azure CLI](https://github.com/Azure/azure-cli).

**Key Features:**

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Intuitive Command Structure:** Uses a consistent `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:** Provides tab completion for commands and parameters to speed up your workflow.
*   **Powerful Querying:** Utilize JMESPath queries (`--query` parameter) for customized output.
*   **Flexible Output Formatting:** Choose from JSON, table, or TSV formats to suit your needs.
*   **Scripting Friendly:** Returns exit codes for easy integration into scripts.
*   **VS Code Integration:** Enhance your experience with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension, offering features like IntelliSense, snippets, and in-terminal command execution.

## Installation

Get started quickly by following the comprehensive [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli).  Troubleshooting resources are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

Quickly execute Azure CLI commands:

```bash
$ az [ group ] [ subgroup ] [ command ] {parameters}
```

Explore command functionality:

```bash
$ az storage -h
$ az vm create -h
```

## Highlights & Effective Usage

The Azure CLI includes many features to help you effectively manage Azure resources.

![Azure CLI Highlight Reel](doc/assets/AzBlogAnimation4.gif)

### Tab Completion

Use tab completion to speed up command entry:

```bash
$ az vm show -g [tab][tab]
AccountingGroup   RGOne  WebPropertiesRG

$ az vm show -g WebPropertiesRG -n [tab][tab]
StoreVM  Bizlogic

$ az vm show -g WebPropertiesRG -n Bizlogic
```

### Querying

Customize your output using `--query` and JMESPath:

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

Understand command success and failure through standardized exit codes:

| Exit Code | Scenario                                               |
| --------- | ------------------------------------------------------ |
| 0         | Command ran successfully.                             |
| 1         | Generic error (bad status code, CLI validation failed) |
| 2         | Parser error; check input to command line.            |
| 3         | Missing ARM resource.                                  |

### Common Scenarios

Improve your workflows with these common scenarios:

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## Resources

*   **Samples and Snippets:** Explore more usage examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and [Microsoft documentation](https://learn.microsoft.com/cli/azure/overview).

## Developer Builds

*   **Docker:** Use a preconfigured Docker image: See [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list).
*   **Edge Builds:** Access the latest `dev` branch builds:
    *   [Windows](https://aka.ms/InstallAzureCliWindowsEdge)
    *   [Homebrew](https://aka.ms/InstallAzureCliHomebrewEdge)
    *   [Ubuntu Bionic Deb](https://aka.ms/InstallAzureCliBionicEdge)
    *   [Ubuntu Focal Deb](https://aka.ms/InstallAzureCliFocalEdge)
    *   [Ubuntu Jammy Deb](https://aka.ms/InstallAzureCliJammyEdge)
    *   [RPM el8](https://aka.ms/InstallAzureCliRpmEl8Edge)
*   **Arbitrary Builds:** Get builds of specific commits/PRs: [Try new features before release](doc/try_new_features_before_release.md).

## Developer Setup

*   **Contribute:** Learn how to [configure your machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md) and [author command modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules) and [code generation](https://github.com/Azure/aaz-dev-tools).
*   **Contribute Code:** Adhere to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  See the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).

## Reporting Issues and Feedback

*   **Report Bugs:** File issues in the [Issues](https://github.com/Azure/azure-cli/issues) section of the GitHub repo.
*   **Provide Feedback:** Use the `az feedback` command or contact the developer team via azpycli@microsoft.com.

## Data Collection & Telemetry

The Azure CLI collects usage data to improve its products.  Review the [Microsoft privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).  To disable telemetry, run `az config set core.collect_telemetry=no`.