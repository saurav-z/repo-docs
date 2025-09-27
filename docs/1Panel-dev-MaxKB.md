<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

<h1 align="center">MaxKB: Build Enterprise-Grade AI Agents with Ease</h1>

<p align="center">MaxKB empowers you to build powerful, intelligent agents for various enterprise applications, offering a comprehensive open-source solution.</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>    
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
 [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>] 
</p>

<hr/>

**MaxKB (Max Knowledge Brain)** is an open-source platform designed for creating sophisticated, enterprise-grade AI agents.  It leverages Retrieval-Augmented Generation (RAG) pipelines, robust workflows, and advanced tool-use capabilities, making it ideal for intelligent customer service, internal knowledge bases, and more.

**Key Features:**

*   **RAG Pipeline:**
    *   Supports document uploading and web crawling for knowledge ingestion.
    *   Includes automated text splitting and vectorization to optimize performance.
    *   Reduces hallucinations in large language models (LLMs).
*   **Agentic Workflow:**
    *   Features a powerful workflow engine.
    *   Offers a comprehensive function library and MCP tool use.
    *   Enables orchestration of AI processes for complex business needs.
*   **Seamless Integration:**
    *   Allows quick and easy integration into third-party systems with minimal coding.
    *   Enhances existing systems with intelligent Q&A capabilities.
*   **Model-Agnostic:**
    *   Supports various LLMs, including private (e.g., DeepSeek, Llama, Qwen) and public models (e.g., OpenAI, Claude, Gemini).
*   **Multi-Modal Support:**
    *   Native support for text, image, audio, and video input and output.

## Quick Start

Get started with MaxKB using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the web interface at `http://your_server_ip:8080` with default credentials:

*   **Username:** admin
*   **Password:** MaxKB@123..

For users in China experiencing Docker image pull issues, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for an alternative installation method.

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

MaxKB is licensed under the GNU General Public License version 3 (GPLv3).  See the [license](https://www.gnu.org/licenses/gpl-3.0.html) for details.

[Link to Original Repo](https://github.com/1Panel-dev/MaxKB)
```

Key improvements and SEO considerations:

*   **Headline Optimization:**  Uses a strong, keyword-rich headline: "MaxKB: Build Enterprise-Grade AI Agents with Ease."  This targets the primary use case and benefit.
*   **Concise Hook:** A one-sentence hook to immediately grab the reader's attention.
*   **Clear Sectioning:** Uses headings (H1, H2) to structure the content, making it easy to scan and improving SEO by indicating the importance of topics.
*   **Bulleted Key Features:** Uses bullet points for readability and to highlight important features. This format is search-engine-friendly.
*   **Keyword Integration:** Naturally includes relevant keywords like "AI agents," "enterprise-grade," "RAG pipeline," "open-source," and model-related terms (OpenAI, etc.).
*   **Concise Descriptions:**  Rephrases sentences for greater clarity and impact.
*   **Clear Call to Action:** The Quick Start section provides immediate value and encourages users to try the software.
*   **Technical Stack and License Sections:**  Includes relevant information for developers and users.
*   **Link to Original Repo:** Added a clear link back to the original repository as requested.
*   **Screenshots:** Retained screenshots to visually demonstrate the platform.
*   **SEO-friendly Formatting:**  Uses Markdown formatting for headings, lists, and bold text, making the document easily parsable by search engines.