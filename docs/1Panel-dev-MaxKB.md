<p align="center">
  <img src="https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" />
</p>

<h1 align="center">MaxKB: Your Open-Source Platform for Enterprise-Grade AI Agents</h1>

MaxKB empowers you to build cutting-edge, intelligent AI agents for a variety of enterprise applications.  ([View the original repository](https://github.com/1Panel-dev/MaxKB))

<p align="center">
  <a href="https://trendshift.io/repositories/9113" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9113" alt="1Panel-dev%2FMaxKB | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>]
</p>
<hr/>

## Key Features

MaxKB is designed to provide a robust and flexible platform for building intelligent agents. Key features include:

*   **Advanced RAG Pipelines:**  Seamlessly integrate Retrieval-Augmented Generation (RAG) pipelines by direct document uploads or web crawling, with automated text splitting, and vectorization for superior Q&A experiences.
*   **Agentic Workflow Engine:** Orchestrate complex AI processes with a powerful workflow engine, function library, and MCP tool-use capabilities.
*   **Rapid Integration:** Quickly equip existing systems with intelligent Q&A capabilities through zero-coding integration, enhancing user satisfaction.
*   **Model Agnostic:** Supports various Large Language Models (LLMs), including private models (DeepSeek, Llama, Qwen, etc.) and public models (OpenAI, Claude, Gemini, etc.).
*   **Multi-Modal Support:**  Native support for text, image, audio, and video input and output, enabling richer interactions.

## Getting Started

### Quick Start with Docker

Get up and running quickly with the following Docker command:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` using the default credentials:

*   **Username:** admin
*   **Password:** MaxKB@123..

**Note for Chinese Users:**  If you encounter issues pulling the Docker image, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/).

## Screenshots

<table style="border-collapse: collapse; border: 1px solid black;">
  <tr>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/user-attachments/assets/eb285512-a66a-4752-8941-c65ed1592238" alt="MaxKB Demo1"   /></td>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/user-attachments/assets/f732f1f5-472c-4fd2-93c1-a277eda83d04" alt="MaxKB Demo2"   /></td>
  </tr>
  <tr>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/user-attachments/assets/c927474a-9a23-4830-822f-5db26025c9b2" alt="MaxKB Demo3"   /></td>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/user-attachments/assets/e6268996-a46d-4e58-9f30-31139df78ad2" alt="MaxKB Demo4"   /></td>
  </tr>
</table>

## Technical Stack

MaxKB is built using a modern technology stack:

*   **Frontend:** [Vue.js](https://vuejs.org/)
*   **Backend:** [Python / Django](https://www.djangoproject.com/)
*   **LLM Framework:** [LangChain](https://www.langchain.com/)
*   **Database:** [PostgreSQL + pgvector](https://www.postgresql.org/)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1Panel-dev/MaxKB&type=Date)](https://star-history.com/#1Panel-dev/MaxKB&Date)

## License

MaxKB is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:**  The title directly states the product and its value proposition. The opening sentence is a concise hook.
*   **Keywords:**  Includes relevant keywords throughout the text, such as "open-source," "AI agents," "enterprise-grade," "RAG pipelines," "LLMs," "workflow engine," etc.  These are crucial for search engine visibility.
*   **Headings and Structure:**  Uses clear headings and subheadings to improve readability and organization.  This also helps search engines understand the content.
*   **Bulleted Lists:**  Highlights key features with bullet points, making them easy to scan and understand.
*   **Action-Oriented Language:**  Uses phrases like "Empowers you," "Get up and running," and "Quickly equip" to encourage user action.
*   **Concise Descriptions:** Keeps descriptions of features brief and to the point.
*   **Links to Relevant Resources:** Includes links to the project's documentation, and licenses, enhancing the user experience and potentially improving SEO.
*   **Alt Text:**  Ensures all images have descriptive alt text, which is critical for accessibility and can also help with SEO.
*   **Simplified Quick Start:** Provides a straightforward and easy-to-copy Docker command.
*   **Clear Call to Action:**  Encourages the reader to visit the web interface after deployment.
*   **Removed Chinese Language in the Title:** Maintained a consistent language choice for readability, and moved the localized text to a supporting section.