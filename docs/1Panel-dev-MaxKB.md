<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

## MaxKB: Build Powerful Enterprise AI Agents with Ease

MaxKB empowers you to build and deploy sophisticated, enterprise-grade AI agents, offering a robust and flexible platform for a variety of use cases.  Learn more and explore the code on the [original repository](https://github.com/1Panel-dev/MaxKB).

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>    
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
 [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>] 
</p>

### Key Features

*   **RAG Pipeline:**  Effortlessly integrate Retrieval-Augmented Generation (RAG) with document uploading, automatic text splitting, and vectorization to reduce LLM hallucinations and improve Q&A accuracy.
*   **Agentic Workflow:** Utilize a powerful workflow engine, function library, and MCP tool-use to orchestrate complex AI processes for sophisticated enterprise solutions.
*   **Seamless Integration:** Quickly equip existing systems with intelligent Q&A capabilities, enhancing user satisfaction with zero-coding integration into third-party systems.
*   **Model-Agnostic:**  Supports a wide range of Large Language Models (LLMs), including both private (e.g., DeepSeek, Llama, Qwen) and public models (e.g., OpenAI, Claude, Gemini).
*   **Multi-Modal Support:** Native support for input and output in text, image, audio, and video formats, enabling rich and engaging interactions.

### Quick Start with Docker

Get started quickly with MaxKB using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` using the following default credentials:

*   Username: `admin`
*   Password: `MaxKB@123..`

**For Chinese users encountering Docker image pull issues, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for alternative installation methods.**

### Screenshots

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

### Technical Stack

*   **Frontend:** [Vue.js](https://vuejs.org/)
*   **Backend:** [Python / Django](https://www.djangoproject.com/)
*   **LLM Framework:** [LangChain](https://www.langchain.com/)
*   **Database:** [PostgreSQL + pgvector](https://www.postgresql.org/)

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1Panel-dev/MaxKB&type=Date)](https://star-history.com/#1Panel-dev/MaxKB&Date)

### License

Licensed under The GNU General Public License version 3 (GPLv3). You can find the full license text at:

<https://www.gnu.org/licenses/gpl-3.0.html>
```
Key improvements and SEO optimizations:

*   **Clear Title & Hook:**  The title now immediately states the core benefit ("Build Powerful Enterprise AI Agents") and includes a concise, attention-grabbing hook sentence.
*   **SEO Keywords:** Included relevant keywords throughout the summary: "enterprise AI agents", "RAG pipeline", "LLM", "workflow engine", etc.  The title and headings are optimized for search.
*   **Concise Feature Descriptions:**  Key features are bulleted for easy scanning and quick understanding. Descriptions are short, benefit-oriented, and use strong action verbs.
*   **Structured Formatting:** Headings and subheadings are used to improve readability and help search engines understand the document's structure.
*   **Call to Action:**  The "Quick Start" section provides a clear and immediate way for users to engage with the project.
*   **Link Back to Original Repo:** Added a prominent link to the original repository for direct access.
*   **Simplified Language:** Replaced some of the original wording with more direct and active language.
*   **Removed Redundancy:** Streamlined the introduction by directly stating MaxKB's purpose and core benefit.
*   **Markdown Formatting:**  Used standard markdown for easy rendering on various platforms (e.g., GitHub, GitLab).
*   **Emphasis on Benefits:** Features are described in terms of what they *do* for the user, rather than just listing technical details.