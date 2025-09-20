<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

# MaxKB: Build Powerful AI Agents for Your Enterprise (Open Source)

**MaxKB empowers you to build cutting-edge, enterprise-grade AI agents with its open-source platform.**

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>    
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>]
</p>
<hr/>

MaxKB (Max Knowledge Brain) is an open-source platform designed to streamline the development of sophisticated, enterprise-ready AI agents. It offers a robust and user-friendly environment for building intelligent solutions across various applications.

**Key Features:**

*   **RAG Pipeline:** Effortlessly ingest documents (direct upload or web crawling) with automated text splitting and vectorization, dramatically improving Large Language Model (LLM) accuracy and reducing hallucinations for superior Q&A experiences.
*   **Agentic Workflow:** Leverage a powerful workflow engine, function library, and MCP tool-use capabilities to orchestrate complex AI processes, perfectly tailored to meet intricate business demands.
*   **Seamless Integration:**  Integrate effortlessly with existing third-party systems without requiring extensive coding, quickly adding intelligent Q&A capabilities to boost user satisfaction.
*   **Model Agnostic:**  Supports a wide range of LLMs, including private models (DeepSeek, Llama, Qwen, etc.) and public models (OpenAI, Claude, Gemini, etc.).  Choose the model that best fits your needs.
*   **Multi-Modal Support:** Works natively with text, image, audio, and video inputs and outputs.

## Quick Start with Docker

Get up and running quickly using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` with the following default credentials:

*   username: `admin`
*   password: `MaxKB@123..`

**Note for Chinese Users:** If you encounter issues pulling the Docker image, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for alternative installation methods.

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

MaxKB is licensed under the GNU General Public License v3 (GPLv3).  See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.html) for details.

**[Back to the MaxKB Repository](https://github.com/1Panel-dev/MaxKB)**
```
Key improvements and SEO considerations:

*   **Headline Optimization:**  The primary headline is now more keyword-rich ("Build Powerful AI Agents") and includes a benefit ("for Your Enterprise") and the keyword "Open Source" to attract users searching for open-source AI tools.
*   **Concise Introduction:**  A single, compelling sentence immediately grabs the reader's attention.
*   **Feature Highlighting with Bullets:**  Key features are clearly presented using bullet points, making them easy to scan. Each bullet uses benefit-driven language.
*   **Stronger Call to Action:**  "Get up and running quickly" encourages immediate action in the Quick Start section.
*   **Clear Headings:**  Consistent and clear headings guide the reader.
*   **SEO Keywords:**  The text incorporates relevant keywords like "AI Agents," "Enterprise AI," "Open Source," "RAG," and LLM" throughout the description.
*   **Contextual Links:**  All links are retained and are now properly formatted. Includes a link back to the original repository.
*   **Readability:** Improved formatting, spacing, and sentence structure enhance readability.
*   **License Information:** Clearly states the license.
*   **Screenshots:** Screenshots are retained for visual appeal.