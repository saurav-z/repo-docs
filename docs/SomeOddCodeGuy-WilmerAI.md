# WilmerAI: Orchestrate LLMs with Advanced Contextual Routing and Node-Based Workflows

**Unlock the power of advanced language model orchestration with WilmerAI, enabling smarter conversations and complex task automation.** ([Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features

*   **Advanced Contextual Routing:** Directs user requests using sophisticated, context-aware logic.  WilmerAI analyzes the entire conversation history to understand user intent, not just the last message.

*   **Core: Node-Based Workflow Engine:**  Build complex, multi-step processes using JSON-defined workflows. Each node can execute different LLMs, call external tools, run custom scripts, and more.

*   **Multi-LLM & Multi-Tool Orchestration:**  Orchestrate the best model for each task component. Workflows can combine multiple LLMs and external tools for optimal performance.

*   **Modular & Reusable Workflows:** Create self-contained workflows for common tasks and use them as building blocks in larger, more complex workflows.

*   **Stateful Conversation Memory:** Maintains long-term context with a chronological summary, a continuously updated rolling summary, and a searchable vector database for Retrieval-Augmented Generation (RAG).

*   **Adaptable API Gateway:**  Exposes OpenAI- and Ollama-compatible API endpoints, enabling easy integration with existing front-end tools.

*   **Flexible Backend Connectors:** Connect to various LLM backends (OpenAI, Ollama, KoboldCpp) using a configurable system of Endpoints, API Types, and Presets.

*   **MCP Server Tool Integration using MCPO:** Experimental support for MCP server tool calling using MCPO, allowing tool use mid-workflow.

## Overview

WilmerAI is designed for advanced semantic prompt routing and task orchestration. It goes beyond simple keyword-based routing by analyzing the *entire conversation history* to determine user intent. The core of WilmerAI is a **node-based workflow engine**, allowing you to define complex, multi-step processes that leverage multiple LLMs, external tools, and custom scripts. This approach provides granular control over the LLM interaction process, improving performance and flexibility.

## Example Use Cases

See how WilmerAI can be used to create complex workflows with multiple LLMs.

### Semi-Autonomous Workflows

![No-RAG vs RAG](Docs/Gifs/Search-Gif.gif)
*Click the image to play gif if it doesn't start automatically*

### Iterative LLM Calls To Improve Performance

Iterate the results of an LLM call to get better results

### Distributed LLMs

Orchestrate multiple machines running LLMs to create more powerful workflows.

### Example Visuals

*   **Example of A Simple Assistant Workflow Using the Prompt Router**

    ![Single Assistant Routing to Multiple LLMs](Docs/Examples/Images/Wilmer-Assistant-Workflow-Example.jpg)

*   **Example of How Routing Might Be Used**

    ![Prompt Routing Example](Docs/Examples/Images/Wilmer-Categorization-Workflow-Example.png)

*   **Group Chat to Different LLMs**

    ![Groupchat to Different LLMs](Docs/Examples/Images/Wilmer-Groupchat-Workflow-Example.png)

*   **Example of a UX Workflow Where A User Asks for a Website**

    ![Oversimplified Example Coding Workflow](Docs/Examples/Images/Wilmer-Simple-Coding-Workflow-Example.jpg)

## Quick Setup

### Youtube Videos

[![WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop](https://img.youtube.com/vi/KDpbxHMXmTs/0.jpg)](https://www.youtube.com/watch?v=KDpbxHMXmTs "WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop")

### Guides

#### WilmerAI

Hop into the [User Documents Setup Starting Guide](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md) to get
step by step rundown of how to quickly set up the API.

#### Wilmer with Open WebUI

[You can click here to find a written guide for setting up Wilmer with Open WebUI](Docs/_User_Documentation/Setup/Open-WebUI.md)

#### Wilmer With SillyTavern

[You can click here to find a written guide for setting up Wilmer with SillyTavern](Docs/_User_Documentation/Setup/SillyTavern.md).

## Documentation

*   [User Documentation](Docs/_User_Documentation/README.md)
*   [Developer Documentation](Docs/Developer_Docs/README.md)

## Contact

For feedback, requests, or just to say hi, you can reach me at:

WilmerAI.Project@gmail.com

## Third-Party Libraries

WilmerAI imports five libraries within its requirements.txt, and imports the libraries via import statements; it does
not extend or modify the source of those libraries.

The libraries are:

*   Flask : https://github.com/pallets/flask/
*   requests: https://github.com/psf/requests/
*   scikit-learn: https://github.com/scikit-learn/scikit-learn/
*   urllib3: https://github.com/urllib3/urllib3/
*   jinja2: https://github.com/pallets/jinja
*   pillow: https://github.com/python-pillow/Pillow

Further information on their licensing can be found within the README of the ThirdParty-Licenses folder, as well as the
full text of each license and their NOTICE files, if applicable, with relevant last updated dates for each.

## Wilmer License and Copyright

```
WilmerAI
Copyright (C) 2025 Christopher Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```