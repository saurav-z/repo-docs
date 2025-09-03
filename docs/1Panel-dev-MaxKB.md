<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c06949962-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

## MaxKB: Build Powerful Enterprise-Grade AI Agents with Ease

MaxKB is an open-source platform designed to empower you to build intelligent, enterprise-ready AI agents quickly and efficiently.  ([See the original repo](https://github.com/1Panel-dev/MaxKB))

<p align="center"><a href="https://trendshift.io/repositories/9113" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9113" alt="1Panel-dev%2FMaxKB | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>
<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>    
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
 [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>] 
</p>
<hr/>

### Key Features

*   **RAG Pipeline:** Leverage Retrieval-Augmented Generation (RAG) pipelines for enhanced accuracy.  Supports document uploads, web crawling, automatic text splitting, and vectorization to minimize Large Language Model (LLM) hallucinations and provide superior Q&A experiences.
*   **Agentic Workflow:** Create sophisticated AI processes with a powerful workflow engine, a rich function library, and MCP tool-use capabilities, enabling the automation of complex tasks.
*   **Seamless Integration:** Quickly integrate MaxKB into your existing systems without coding, adding intelligent Q&A functionality to improve user satisfaction.
*   **Model-Agnostic:** Compatible with a wide range of LLMs, including private models (DeepSeek, Llama, Qwen, etc.) and public models (OpenAI, Claude, Gemini, etc.).
*   **Multi-Modal Support:**  Native support for text, image, audio, and video inputs and outputs.

### Quick Start with Docker

Get up and running quickly with the following Docker command:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` using the default credentials:

*   username: admin
*   password: MaxKB@123..

**For Chinese users experiencing Docker image pull issues**, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/).

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

*   **Frontend:** Vue.js ([https://vuejs.org/](https://vuejs.org/))
*   **Backend:** Python / Django ([https://www.djangoproject.com/](https://www.djangoproject.com/))
*   **LLM Framework:** LangChain ([https://www.langchain.com/](https://www.langchain.com/))
*   **Database:** PostgreSQL + pgvector ([https://www.postgresql.org/](https://www.postgresql.org/))

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1Panel-dev/MaxKB&type=Date)](https://star-history.com/#1Panel-dev/MaxKB&Date)

### License

MaxKB is licensed under the GNU General Public License version 3 (GPLv3).  See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.html) for details.
```

Key improvements:

*   **SEO-Optimized Title:**  Uses "MaxKB" as the primary keyword and highlights key benefits.
*   **Concise Hook:**  Provides a clear, one-sentence overview.
*   **Structured Headings:**  Uses clear headings (Key Features, Quick Start, etc.) for readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points for easy scanning and comprehension.  Each feature description is more concise.
*   **Clear Call to Action:**  Encourages users to try the product.
*   **Technical Stack & License:**  Includes these important details.
*   **Links:** Adds links where appropriate (repo link, license).
*   **Concise Language:**  Rewords the original text for better flow and clarity.
*   **Keywords:** Includes relevant keywords like "AI agents," "RAG," "LLM," "enterprise," and key technologies.