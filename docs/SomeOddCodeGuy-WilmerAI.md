# WilmerAI: Expertly Routing Language Models for Enhanced AI Interactions

**Supercharge your AI assistant by orchestrating multiple LLMs to work together, delivering richer and more dynamic responses.** [Explore the Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Intelligent Prompt Routing:** Direct prompts to specific domains (coding, factual, etc.) or personas for tailored responses.
*   **Customizable Workflows:** Design workflows that leverage multiple LLMs in sequence or parallel, maximizing model capabilities.
*   **Multi-LLM Orchestration:** Combine responses from numerous LLMs, even across different machines, for comprehensive output.
*   **RAG Integration with Wikipedia:** Seamlessly incorporate the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) to enhance factual accuracy.
*   **Dynamic Chat Summaries:** Generate ongoing conversation summaries to enable long-context memory and consistency.
*   **VRAM Optimization with Ollama:** Leverage Ollama's model hotswapping to run complex workflows even on systems with limited VRAM.
*   **Flexible Presets:** Tailor LLM settings through customizable JSON files, enabling seamless adaptation to new models and APIs.
*   **Multi-Modal Support (Experimental):** Process and incorporate images into conversations using Ollama and its vision capabilities.
*   **Mid-Workflow Conditional Logic:** Create dynamic flows that adapt based on the LLM's responses, directing prompts to different workflows.
*   **MCP Server Tool Integration:** Enhanced Tool Calling via MCPO using MCP server tool integration using MCPO, allowing tool use mid-workflow. Big thank you to [iSevenDays](https://github.com/iSevenDays) for the amazing work on this feature.

## What is WilmerAI?

WilmerAI is a powerful intermediary that sits between your applications (like Open WebUI, SillyTavern, or custom Python scripts) and your Language Model APIs. Wilmer allows you to construct custom workflows that can route prompts to multiple LLMs across various computers, and get back a unified result. It offers extensible functionality with custom Python scripts and supports both OpenAI and Ollama compatible endpoints, enabling integration with a wide array of tools.

## Important Notices

**Disclaimer**: WilmerAI is a personal project under active development and is provided "as-is" without any warranty.

**Maintainer's Note (Updated 2025-06-29)**: Significant improvements have been made to how Wilmer handles reasoning models, with features for stripping out thinking blocks. See the endpoint configs section for more. Due to work commitments, major modifications to Middleware modules will not be accepted until September 2025. The project is open for contributions to iSevenDays' MCP tool calling feature, adding new users, or prompt templates.

## Quick Setup Guide

WilmerAI uses JSON configuration files located in the "Public" folder.
*   **Step 1:** Install Python and the dependencies (`pip install -r requirements.txt`).
*   **Step 2:** Update Endpoints configurations for your LLMs in `Public/Configs/Endpoints`.
*   **Step 3:** Select the desired user by editing `Public/Configs/Users/_current-user.json`.
*   **Step 4:** Configure your routes in the `Routing` folder.
*   **Step 5:** Ensure your new user's configuration specifies the correct routing and associated Workflows.

## Understanding the Key Concepts

### Workflows

Workflows are the core of WilmerAI's functionality. They define the sequence and logic for interacting with LLMs.

### Endpoints

Define the connection details for each LLM API, including type (OpenAI, Ollama, etc.), URL, and model information.

### Prompt Templates

These templates ensure that prompts are correctly formatted for each LLM, maximizing compatibility and performance.

### Users

User configurations dictate the specific settings, workflows, and endpoints to be used for different applications or use
cases.

### Routing

Routing configuration directs prompts to specific workflows based on categories or other criteria.

## Detailed Sections (Summarized)

### Endpoints

Define your LLM API connections, specifying API type, context sizes, model names, and prompt templates.

### ApiTypes

Configuration files representing the different API types supported by Wilmer.

### PromptTemplates

Prompt templates format prompts for each LLM, increasing compatibility.

### Creating a User

*   **Users Folder:** Create a JSON file for each user defining its specific settings and the other configuration files it
    requires.
*   **Users Folder, _current-user.json File:** Specifies which user configuration to load.
*   **Routing Folder:** Define how prompts are categorized and routed to workflows.
*   **Workflow Folder:** Create a folder to contain the workflows tied to your User.

### Understanding Workflows

Workflows consist of nodes that define the flow of execution.

### Node Properties

Each node has the properties to make calls to the LLMs and how they function.

### Types of Nodes

WilmerAI supports various node types, including:

*   **Regular Chat Nodes:** These nodes simply send prompts to an LLM and return a response.
*   **Recent Memory Summarizer Tool:** This node processes the last N messages to create summaries.
*   **Recent/Quality Memory Node:** This node will do a keyword search against the memories for anything related to the current
    conversation.
*   **Full Chat Summary Node:** Combines recent memories into a comprehensive summary.
*   **Get Current Chat Summary From File** Get's the current chat summary file.
*   **Parallel Processing Node:** Utilizes multiple LLMs to process memories and summaries in parallel.
*   **Python Module Caller Node:** Executes custom Python scripts for extended functionality.
*   **Full/Partial Text Wikipedia Offline API Caller Node:** Integrates with the Offline Wikipedia API for factual lookups.
*   **Get Custom File:** Loads a custom text file in a workflow.
*   **Workflow Lock:** Synchronizes asynchronous operations.
*   **Image Processor:** Captions images sent to the backend.
*   **Custom Workflow Node:** Executes sub-workflows within a workflow.
*   **Conditional Custom Workflow Node:** Executes workflows based on a condition.

### Presets

Presets offer flexible and customizable configurations for LLM parameters.

## Troubleshooting

*   **Memories/Summaries Missing:** Check that the nodes, files and their paths are properly configured.
*   **Front-End Issues:** Ensure streaming matches between WilmerAI and your front end and correct user selected.
*   **Preset Errors:** Update presets to be compatible with the specific LLM APIs being used.
*   **Out of Memory/Truncation Errors:** Ensure your prompt and response size token limits are set properly.
*   **API Failures:** Inspect the output for details on LLM API errors.
*   **No Response:** Check that the correct user is set and your endpoints and workflow are configured.

## Contact

For support and feedback, contact WilmerAI.Project@gmail.com.

## License and Third-Party Libraries

WilmerAI is licensed under the GNU General Public License v3. The project uses several open-source libraries, with details on
licensing available in the README in the ThirdParty-Licenses folder.