# Azure CLI: Manage Your Azure Resources from the Command Line

**Effortlessly manage and automate your Azure cloud resources with the powerful and versatile Azure CLI.**  For more information and to contribute, check out the original repository: [Azure CLI](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform:**  Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive:** Supports a vast range of Azure services and resources.
*   **Scriptable:** Enables automation through scripting and integration with other tools.
*   **Intuitive Command Structure:** Uses a clear `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:** Provides tab completion for commands, groups, and parameters, speeding up command entry.
*   **Flexible Output:** Supports JSON, table, and TSV output formats for easy data consumption.
*   **Powerful Querying:**  Utilizes JMESPath for custom output formatting and data extraction.
*   **Extensible:**  Supports custom extensions and modules for enhanced functionality.

## Get Started

*   **Installation:** Follow the detailed [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli).
*   **Quick Start:**  Explore the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   **Help:** Get help for any command using the `-h` parameter (e.g., `az storage -h`).

## Core Concepts & Highlights

*   **Tab Completion:**  Increase efficiency with auto-completion.
*   **Querying with JMESPath:** Customize your output with the `--query` parameter for precise data retrieval.
*   **Exit Codes:** Understand and leverage exit codes for reliable scripting.

## Common Scenarios & Effective Usage

Explore common use cases and best practices:

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## More Resources

*   **Samples & Snippets:**  Explore extensive examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and at [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).
*   **VS Code Integration:** Enhance your development workflow with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for Visual Studio Code, including features like IntelliSense, snippets, and integrated terminal execution.

## Telemetry & Reporting Issues

*   **Telemetry:** Data collection is enabled by default. Opt out with `az config set core.collect_telemetry=no`.
*   **Feedback:**  Use the `az feedback` command or file issues in the [Issues](https://github.com/Azure/azure-cli/issues) section of the GitHub repo.

## Developer Information

*   **Docker:** Utilize preconfigured Docker images for easy access.
*   **Edge Builds:** Access the latest development builds.
*   **Developer Setup:**  Learn how to set up a development environment and contribute to the project.