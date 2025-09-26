<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

<h1 align="center">MaxKB: Build Enterprise-Grade AI Agents with Ease</h1>

MaxKB is the open-source platform designed to empower you to build intelligent agents for your enterprise needs.  (See the original repo [here](https://github.com/1Panel-dev/MaxKB).)

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>]
</p>

<hr/>

MaxKB, or Max Knowledge Brain, provides a comprehensive platform for creating and deploying AI agents within your organization.  It seamlessly integrates Retrieval-Augmented Generation (RAG) pipelines, robust workflows, and advanced tool-use capabilities to address complex enterprise challenges.

**Key Features:**

*   **RAG Pipeline:** Effortlessly ingest documents and automatically crawl online content with built-in text splitting and vectorization, significantly improving the accuracy and reliability of your AI interactions.
*   **Agentic Workflow:** Orchestrate complex AI processes with a powerful workflow engine, extensive function library, and MCP tool-use, enabling sophisticated task automation.
*   **Seamless Integration:** Quickly integrate MaxKB into your existing systems with zero-code solutions, adding intelligent Q&A capabilities to enhance user satisfaction.
*   **Model-Agnostic:** Compatible with a wide range of large language models (LLMs), including both private models (e.g., DeepSeek, Llama, Qwen) and public models (e.g., OpenAI, Claude, Gemini).
*   **Multi-Modal Support:** Native support for diverse input and output formats, including text, images, audio, and video.

## Quick Start with Docker

Get started quickly by running MaxKB in a Docker container:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` with the default admin credentials:

*   **Username:** admin
*   **Password:** MaxKB@123..

For Chinese users experiencing Docker image pull issues, refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/).

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

*   **Frontend:** [Vue.js](https://vuejs.org/)
*   **Backend:** [Python / Django](https://www.djangoproject.com/)
*   **LLM Framework:** [LangChain](https://www.langchain.com/)
*   **Database:** [PostgreSQL + pgvector](https://www.postgresql.org/)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1Panel-dev/MaxKB&type=Date)](https://star-history.com/#1Panel-dev/MaxKB&Date)

## License

Licensed under The GNU General Public License version 3 (GPLv3).  You can find more details at:

<https://www.gnu.org/licenses/gpl-3.0.html>

The software is provided "AS IS" without warranties or conditions of any kind.  Please refer to the license for specific permissions and limitations.
```
Key improvements and explanations:

*   **SEO-Optimized Title and Description:**  The title is now more keyword-rich ("Build Enterprise-Grade AI Agents") and includes the primary keyword "AI Agents".  The one-sentence hook highlights the platform's core value proposition.
*   **Clear Headings:** Uses `<h1>` for the main title and `<h2>` for sections, improving readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points to clearly present the core functionalities.
*   **Concise and Engaging Language:**  The text is rewritten for clarity and conciseness.
*   **Call to Action:** Encourages the user to learn more by including the GitHub link.
*   **Simplified Quick Start:** Remains the same, as that's a functional aspect.
*   **Complete Technical Stack:** Highlights the technology used.
*   **Clear Licensing Information:**  Explains the license terms.
*   **Removed redundant phrases:** Removed phrases like "MaxKB = Max Knowledge Brain" and directly introduced the platform.
*   **Focus on Benefits:** The "Key Features" section focuses on the *benefits* of using MaxKB (e.g., "improving the accuracy and reliability of your AI interactions") rather than just listing features.