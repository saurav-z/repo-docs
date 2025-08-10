# WilmerAI: Expertly Routing Language Models for Enhanced AI Workflows

**Unleash the power of interconnected LLMs with WilmerAI, a flexible application that orchestrates complex workflows, providing dynamic prompt routing, custom model integrations, and extended memory capabilities.** [Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features:

*   **Intelligent Prompt Routing**: Direct prompts to specific categories or personas, leveraging user-defined workflows for optimal results.
*   **Customizable Workflows**: Design intricate sequences with multiple LLMs and tools, tailoring the AI's approach for diverse tasks.
*   **Multi-LLM Collaboration**: Enable a single prompt to be processed by several LLMs simultaneously, creating sophisticated, collaborative responses.
*   **Offline Wikipedia Integration**: Utilize the OfflineWikipediaTextApi for enhanced factual accuracy in responses.
*   **Persistent Conversation Memory**: Generate and maintain continuous chat summaries, enabling LLMs to maintain context beyond their typical limits.
*   **Model Hotswapping via Ollama**: Maximize VRAM utilization by hotswapping LLMs via Ollama, allowing complex workflows on resource-constrained systems.
*   **Configurable Presets**: Easily customize LLM parameters using JSON presets, adapting to new models and samplers as they become available.
*   **Vision Multi-Modal Support**:  Leverage image processing with Ollama for detailed image analysis and integration into conversations.
*   **Mid-Workflow Conditional Routing**: Dynamically switch between sub-workflows based on conditions for adaptive responses.
*   **MCPO Server Tool Integration**: Experiment with MCP server tool calling using MCPO to utilize tools mid-workflow.

## Get Started:

WilmerAI is designed for flexibility and customization.  Configuration is handled through JSON files.

*   **Installation**:  Install Python and run the included `.bat` (Windows) or `.sh` (macOS) script, or install dependencies manually using `pip install -r requirements.txt` and then `python server.py`.
*   **Configuration**: Modify settings in the `Public` directory for endpoints, routing, and user profiles.  See the provided YouTube videos for helpful setup guidance.
*   **Integration**: Connect your front-end application to WilmerAI's OpenAI or Ollama compatible endpoints.

##  Understanding WilmerAI (and some warnings)

Wilmer was built in the Llama 2 era to solve the need for fine-tune routing. Today, the focus has become semi-autonomous workflows that give the user more control of the path LLMs take.

**IMPORTANT NOTES**
*   A) Preset files are 100% customizable. What is in that file goes to the llm API. This is because cloud APIs do not handle some of the various presets that local LLM APIs handle. As such, if you use OpenAI API or other cloud services, the calls will probably fail if you use one of the regular local AI presets. Please see the preset "OpenAI-API" for an example of what openAI accepts.
*   B) All prompts in Wilmer now use third person.
*   C) By default, all the user files are set to turn on streaming responses. You either need to enable this in your front end that is calling Wilmer so that both match, or you need to go into Users/username.json and set Stream to "false". If you have a mismatch, where the front end does/does not expect streaming and your wilmer expects the opposite, nothing will likely show on the front end.

**Important Considerations**:

*   **Token Usage**:  WilmerAI does not track token usage. Monitor your LLM API dashboards for cost management.
*   **LLM Quality**: The output quality is heavily reliant on the connected LLMs. The quality of your presets and prompt templates are also important.
*   **Disclaimer:** This is a project under development and is provided "as-is" without any warranty.

## Additional Resources:

*   **Setup Tutorial**:  [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
*   **Tutorial Playlist**: [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)
*   **Connecting to SillyTavern**: Connect as OpenAI Compatible v1/Completions or Ollama api/generate endpoints, using provided SillyTavern templates for optimal context.

## Contact

For feedback, requests, or just to say hi, you can reach me at:

WilmerAI.Project@gmail.com

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

Further information on their licensing can be found within the README of the ThirdParty-Licenses folder, as well as the
full text of each license and their NOTICE files, if applicable, with relevant last updated dates for each.

## Wilmer License and Copyright

    WilmerAI
    Copyright (C) 2024 Christopher Smith

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
Key improvements and SEO:

*   **Concise and Engaging Title & Hook:** The title clearly states the project's core function, followed by a punchy one-sentence description.
*   **Structured Headings:**  Uses clear, descriptive headings to organize information, improving readability and SEO.
*   **Bulleted Key Features:**  Highlights the main selling points in a concise, scannable format.
*   **SEO Keywords:**  Incorporates relevant keywords like "LLM", "Workflows", "Prompt Routing", "AI", etc.
*   **Call to Action:** Includes a clear "Get Started" section guiding users.
*   **Clear "How-to-Get-Started" Section**:  Provides a simplified getting started experience that's less complex than the original README.
*   **Focused Content:** Removes some of the more specific technical documentation from the main README, making it more focused on the user's initial engagement.  The longer original README is more of a reference now.
*   **Concise explanations** Rather than a large amount of information, this has a good introduction to the project and the most important details.
*   **License and Copyright Information**: Includes the license and copyright information as per original document.
*   **Clear Contact Information**.
*   **Third party libraries with links and contact info**
*   **Link back to original repo** so that users can reference the original documents