# WilmerAI: Expert Contextual Routing for LLM Workflows

**Unlock advanced, multi-step workflows with WilmerAI, designed to understand context and orchestrate LLMs for superior results.**

[Link to Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Advanced Contextual Routing:**
    *   Analyzes the entire conversation history, not just the latest message.
    *   Employs both prompt routing and in-workflow logic for dynamic task selection.
    *   Enables a deep understanding of user intent for more accurate responses.

*   **Node-Based Workflow Engine:**
    *   Builds complex workflows using JSON-defined sequences of "nodes."
    *   Each node can orchestrate different LLMs, call external tools, and execute custom scripts.
    *   Facilitates a modular and scalable approach to complex tasks.

*   **Multi-LLM & Multi-Tool Orchestration:**
    *   Allows you to connect and coordinate multiple LLMs within a single workflow.
    *   Optimizes for the best model for each part of a task, enhancing accuracy.
    *   Enables the integration of diverse LLMs, tools, and external resources.

*   **Modular & Reusable Workflows:**
    *   Create self-contained workflows for common operations, simplifying overall design.
    *   Execute reusable nodes within larger, more complex workflows.
    *   Promotes efficiency and reduces redundancy in workflow development.

*   **Stateful Conversation Memory:**
    *   Utilizes a multi-faceted memory system for long and accurate conversations.
    *   Includes a chronological summary file, a continuously updated rolling summary, and a vector database.
    *   Provides the necessary context for effective routing and coherent responses.

*   **Adaptable API Gateway:**
    *   Exposes OpenAI- and Ollama-compatible API endpoints.
    *   Connects with existing front-end applications and tools seamlessly.
    *   Simplifies integration with various LLM front-ends.

*   **Flexible Backend Connectors:**
    *   Connects to various LLM backends, including OpenAI, Ollama, and KoboldCpp.
    *   Employs a simple configuration system of Endpoints, API Types, and Presets.
    *   Supports connections to diverse LLM APIs.

*   **MCP Server Tool Integration:**
    *   Supports the use of MCP server tool calling via MCPO.
    *   Offers the integration of tools mid-workflow, enhancing flexibility.
    *   Leverages contributions from iSevenDays.

## Why Choose WilmerAI?

WilmerAI goes beyond simple routing. It's a powerful workflow engine that enables you to create sophisticated, semi-autonomous AI agents. With its advanced contextual understanding, multi-LLM orchestration, and flexible API, WilmerAI empowers you to build intelligent applications that deliver superior results.

## Getting Started

### Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
2.  **Run the Server:** `python server.py` or use the provided `.bat` (Windows) or `.sh` (macOS) scripts.

### Configuration

1.  **Configure Endpoints:** Set up your LLM connections in the `Public/Configs/Endpoints` directory.
2.  **Select a User:** Choose a pre-built user configuration in `Public/Configs/Users` or create your own.

## Connect with Existing Tools

*   **SillyTavern:** Connect as OpenAI Compatible v1/completions or Ollama api/generate. Import the instruct template from `/Docs/SillyTavern/InstructTemplate`
*   **Open WebUI:** Connect to WilmerAI as an Ollama instance.

## Documentation

*   User Documentation: `/Docs/_User_Documentation/README.md`
*   Developer Documentation: `/Docs/Developer_Docs/README.md`

## Important Notes

*   **Token Usage:** Monitor your token usage from your LLM API dashboards, as WilmerAI does not track token consumption.
*   **LLM Dependency:** WilmerAI's quality relies heavily on the performance of the connected LLMs.
*   **Maintainer's Note:** This is an active personal project, with updates being made in free time.

## Contact

For support, feedback, or inquiries: WilmerAI.Project@gmail.com

---

## License and Copyright

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

---

## Third Party Libraries

WilmerAI imports five libraries within its requirements.txt, and imports the libraries via import statements; it does
not extend or modify the source of those libraries.

The libraries are:

* Flask : https://github.com/pallets/flask/
* requests: https://github.com/psf/requests/
* scikit-learn: https://github.com/scikit-learn/scikit-learn/
* urllib3: https://github.com/urllib3/urllib3/
* jinja2: https://github.com/pallets/jinja
* pillow: https://github.com/python-pillow/Pillow

Further information on their licensing can be found within the README of the ThirdParty-Licenses folder, as well as the
full text of each license and their NOTICE files, if applicable, with relevant last updated dates for each.