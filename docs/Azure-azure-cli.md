# Azure CLI: Command-Line Interface for Azure Management

**Simplify and automate your Azure cloud management tasks with the powerful and versatile Microsoft Azure CLI, your cross-platform command-line interface for interacting with Azure.**  ([Back to Original Repo](https://github.com/Azure/azure-cli))

The Azure CLI is a comprehensive command-line tool that allows you to manage and configure Azure resources directly from your terminal.

**Key Features:**

*   **Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.
*   **Resource Management:**  Create, manage, and configure all your Azure services (VMs, Storage, Networking, etc.) from the command line.
*   **Scripting & Automation:**  Easily integrate the Azure CLI into your scripts for automated cloud management.
*   **Tab Completion:**  Save time and reduce errors with tab completion for commands and parameters.
*   **Query Capabilities:**  Use JMESPath queries to customize output and extract specific data.
*   **Output Formatting:**  Choose from various output formats (JSON, table, TSV) to suit your needs.
*   **VS Code Integration:**  Enhance your development workflow with the Azure CLI Tools extension for Visual Studio Code, providing IntelliSense, snippets, and more.
*   **REST API Access:** Utilize `az rest` to directly interact with the Azure REST APIs.
*   **Edge Builds:** Access cutting-edge features and updates with edge builds.

**Installation & Usage:**

*   **Installation:**  Refer to the detailed [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for instructions.
*   **Common Issues:** Find solutions to common installation problems at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).
*   **Basic Syntax:** Use the following format: `$ az [group] [subgroup] [command] {parameters}`
*   **Get Started:**  Explore the [get started guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2) for an in-depth introduction.
*   **Help:** Access help information with the `-h` parameter (e.g., `$ az storage -h`, `$ az vm create -h`).

**Developer Installation:**

*   **Docker:**
    *   We maintain a Docker image preconfigured with the Azure CLI. See our [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list) for available versions.
    *   `$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>`
*   **Edge Builds:** Get the latest features from the `dev` branch.
    *   MSI: https://aka.ms/InstallAzureCliWindowsEdge
    *   Homebrew Formula: https://aka.ms/InstallAzureCliHomebrewEdge
    *   Ubuntu Bionic Deb: https://aka.ms/InstallAzureCliBionicEdge
    *   Ubuntu Focal Deb: https://aka.ms/InstallAzureCliFocalEdge
    *   Ubuntu Jammy Deb: https://aka.ms/InstallAzureCliJammyEdge
    *   RPM el8: https://aka.ms/InstallAzureCliRpmEl8Edge
    *   More details are in the original readme.
*   **Get builds of arbitrary commit or PR:** [Try new features before release](doc/try_new_features_before_release.md)

**Examples & Usage:**

*   **Tab Completion:**
    ```bash
    # looking up resource group and name
    $ az vm show -g [tab][tab]
    AccountingGroup   RGOne  WebPropertiesRG

    $ az vm show -g WebPropertiesRG -n [tab][tab]
    StoreVM  Bizlogic

    $ az vm show -g WebPropertiesRG -n Bizlogic
    ```
*   **Query:**
    ```bash
    $ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
    Name                    Os
    ----------------------  -------
    storevm                 Linux
    bizlogic                Linux
    demo32111vm             Windows
    dcos-master-39DB807E-0  Linux
    ```

**More Information:**

*   **Common Scenarios:**  Explore effective usage scenarios: [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).
*   **Samples:**  Find more examples and snippets at our [GitHub samples repo](http://github.com/Azure/azure-cli-samples) or [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).
*   **VS Code Extension:** [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli)

**Data Collection & Telemetry:**

*   **Telemetry:**  The software collects usage data. You can opt-out using `az config set core.collect_telemetry=no`.
*   **Privacy:**  See the Microsoft privacy statement: https://go.microsoft.com/fwlink/?LinkID=824704.

**Feedback & Contributing:**

*   **Report Issues:** File issues in the [GitHub Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Feedback:**  Use the `az feedback` command from the command line.
*   **Contribution:**  Follow the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) for contributing.
*   **Code of Conduct:**  This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).