# WilmerAI: Revolutionizing Conversational AI with Contextual Routing

**Unlock the power of intelligent, multi-step workflows for advanced semantic prompt routing and complex task orchestration with WilmerAI - a cutting-edge framework for building sophisticated conversational AI experiences.** ([View Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features:

*   **Advanced Contextual Routing:** Understands the full context of a conversation, enabling precise routing based on the entire dialogue history.
*   **Node-Based Workflow Engine:**  Orchestrates complex tasks through customizable workflows defined in JSON, facilitating modularity and reusability.
*   **Multi-LLM & Multi-Tool Integration:**  Leverages multiple LLMs and external tools within a single workflow, optimizing performance and results.
*   **Modular & Reusable Workflows:** Allows for creation of modular workflows for common tasks that can be easily integrated into more complex agents.
*   **Stateful Conversation Memory:**  Maintains context via chronological summaries, rolling summaries, and a vector database for enhanced RAG.
*   **Adaptable API Gateway:** Exposes OpenAI- and Ollama-compatible API endpoints, seamlessly integrating with existing applications.
*   **Flexible Backend Connectors:**  Supports a range of LLM backends (OpenAI, Ollama, KoboldCpp) via a simple, configurable system.
*   **MCPO Server Tool Integration:** Allows tool use mid-workflow.

## Why Use WilmerAI?

WilmerAI empowers you to move beyond basic chatbots, offering:

*   **Improved Accuracy:** By analyzing the entire conversation history, WilmerAI understands user intent more accurately.
*   **Enhanced Capabilities:** Combine multiple LLMs and tools to create advanced, multi-step workflows that address complex tasks.
*   **Seamless Integration:**  Connect WilmerAI to your existing applications with minimal changes.
*   **Customization:**  Fine-tune workflows to match your specific needs and optimize performance.

## Core Concepts and Functionality:

WilmerAI excels at enabling **semi-autonomous Workflows**, giving the user granular control of the path the LLMs take, and allow maximum use of the user's own domain knowledge and experience. With Wilmer's routing, Wilmer routes to many via a whole workflow.

## Example Use Cases:

*   **Advanced Question Answering:**  Leverage context and multiple tools to provide in-depth, accurate responses.
*   **Complex Task Automation:** Automate multi-step processes, such as generating code, summarizing documents, or conducting research.
*   **Intelligent Chatbots:** Create chatbots that understand context, learn from past interactions, and perform sophisticated actions.

## Getting Started:

WilmerAI is easy to set up and use.

1.  **Installation:** Choose from the provided setup scripts or manual installation using pip (see the original README for detailed steps).
2.  **Configuration:** Configure endpoints and user settings using JSON files (found in the "Public" folder).
3.  **Connection:** Connect your front-end application (SillyTavern, Open WebUI, etc.) to WilmerAI's API endpoints.

## API Endpoints:

WilmerAI supports the following API endpoints:

*   OpenAI Compatible v1/completions
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate
*   Ollama Compatible api/chat
*   KoboldCpp Compatible api/v1/generate (non-streaming generate)
*   KoboldCpp Compatible /api/extra/generate/stream (streaming generate)

See the original README for detailed connection instructions for SillyTavern and Open WebUI.

## Documentation:

*   User Documentation: [/Docs/_User_Documentation/README.md](/Docs/_User_Documentation/README.md)
*   Developer Documentation: [/Docs/Developer_Docs/README.md](/Docs/Developer_Docs/README.md)

## Important Considerations:

*   **Token Usage:** WilmerAI does not track or report token usage. Monitor your LLM API dashboards.
*   **Model Quality:**  The quality of WilmerAI's output is directly related to the quality of the connected LLMs and the configuration of your endpoints and prompt templates.

---

**Note:** This project is under active development.