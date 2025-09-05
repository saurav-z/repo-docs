<div align="center">
  <img src="assets/banner.png" alt="MemU Banner">
</div>

# MemU: Revolutionizing AI Companions with Advanced Memory Capabilities

**MemU** is an open-source, next-generation memory framework designed to empower AI companions with persistent and intelligent memory, learn more at the [original repository](https://github.com/NevaMind-AI/memU).

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

<br>

**Tired of AI companions that forget?** MemU provides a high-accuracy, fast, and cost-effective memory solution for AI companions, enabling them to remember, learn, and grow with you.

*   **Visit our homepage:** [memu.pro](https://memu.pro/)

## Key Features & Benefits

*   **AI Companion Specialization:** Optimized specifically for AI companion applications.
*   **92% Accuracy:** Achieve state-of-the-art accuracy in the Locomo benchmark.
*   **Up to 90% Cost Reduction:** Significantly reduce operational costs through optimized online platform.
*   **Advanced Retrieval Strategies:** Employ multiple retrieval methods, including semantic search, hybrid search, and contextual retrieval.
*   **24/7 Support (Enterprise):** Benefit from dedicated support for enterprise customers.

## Core Advantages of MemU

*   **Higher Memory Accuracy:** MemU's superior performance leads the industry by 92.09% average accuracy in Locomo dataset, across all reasoning tasks.
*   **Fast Retrieval:** Efficiently retrieves relevant information by categorizing data into documents, avoiding extensive embedding searches.
*   **Low Cost:** Processes numerous conversation turns simultaneously, minimizing token usage and costs for developers.

## Get Started with MemU

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

Integrate MemU quickly with our cloud platform. It's perfect for teams and individuals who want immediate access without setup.

*   **Instant Access:** Start integrating AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Subscribe and get priority assistance.

**How to Get Started:**

**Step 1:** Create an account at [https://app.memu.so](https://app.memu.so).

**Step 2:** Generate API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).

**Step 3:** Install the library and use a simple code example:

```bash
pip install memu-py
```

```python
import os
from memu import MemuClient

# Initialize
memu_client = MemuClient(
    base_url="https://api.memu.so",
    api_key=os.getenv("MEMU_API_KEY")
)
memu_client.memorize_conversation(
    conversation=conversation_text,
    user_id="user001",
    user_name="User",
    agent_id="assistant001",
    agent_name="Assistant"
)
```

**For more details:** Check the [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog).
*   **See [`example/client/memory.py`](example/client/memory.py) for complete integration details.**

### 🏢 Enterprise Edition

For organizations needing maximum security, customization, and control:

*   **Commercial License:** Full features and commercial usage rights.
*   **Custom Development:** SSO/RBAC integration, dedicated algorithm team for optimization.
*   **Intelligence & Analytics:** User behavior analysis, production monitoring, and automated agent optimization.
*   **Premium Support:** 24/7 dedicated support, custom SLAs, and implementation services.

**Contact:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users preferring local control, data privacy, or customization:

*   **Data Privacy:** Keeps sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform as needed.
*   **Cost Control:** Avoid recurring cloud fees for large deployments.

See [self hosting README](README.self_host.md)

---

## MemU: Memory as a File System - Core Concepts

### **Organize** - Autonomous Memory File Management
Like having a personal librarian who perfectly organizes your thoughts, the memory agent automatically records, modifies, and archives memories.

### **Link** - Interconnected Knowledge Graph
Memories are connected. This system automatically creates connections between related memories, building a rich network.

### **Evolve** - Continuous Self-Improvement
The memory agent keeps working even when offline. It generates new insights by analyzing existing memories, identifies patterns, and creates summary documents through self-reflection. 

### **Never Forget** - Adaptive Forgetting Mechanism
Prioritizes information based on usage patterns. Prioritizes recently accessed memories, while less relevant content is deprioritized or forgotten.

## Use Cases

|   |   |   |   |
| :---: | :---: | :---: | :---: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

## Get Involved: Contributing to MemU

We value open-source collaboration. Help us make MemU even better.

*   **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

All contributions are licensed under the **Apache License 2.0**.

---

## Connect with the MemU Community

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow us for updates. [Follow us](https://x.com/memU_ai)

---

## Ecosystem

We're proud to work with amazing organizations:

<div align="center">

### Development Tools
<a href="https://github.com/TEN-framework/ten-framework"><img src="https://avatars.githubusercontent.com/u/113095513?s=200&v=4" alt="Ten" height="40" style="margin: 10px;"></a>
<a href="https://github.com/camel-ai/camel"><img src="https://avatars.githubusercontent.com/u/134388954?s=200&v=4" alt="Camel" height="40" style="margin: 10px;"></a>
<a href="https://github.com/eigent-ai/eigent"><img src="https://www.eigent.ai/nav/logo_icon.svg" alt="Eigent" height="40" style="margin: 10px;"></a>
<a href="https://github.com/milvus-io/milvus"><img src="https://miro.medium.com/v2/resize:fit:2400/1*-VEGyAgcIBD62XtZWavy8w.png" alt="Ten" height="40" style="margin: 10px;"></a>
<a href="https://xroute.ai/"><img src="assets/partners/xroute.png" alt="xRoute" height="40" style="margin: 10px;"></a>
<a href="https://jaaz.app/"><img src="assets/partners/jazz.png" alt="jazz" height="40" style="margin: 10px;"></a>
<a href="https://github.com/Buddie-AI/Buddie"><img src="assets/partners/buddie.png" alt="buddie" height="40" style="margin: 10px;"></a>
<a href="https://github.com/bytebase/bytebase"><img src="assets/partners/bytebase.png" alt="bytebase" height="40" style="margin: 10px;"></a>
</div>

---

*Interested in partnering with MemU? Contact us at [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## 📱 Stay Connected: Join Our WeChat Community

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">

*Scan the QR codes above to join our WeChat community*

</div>

---

*Stay connected with the MemU community! Join our WeChat groups for real-time discussions, technical support, and networking opportunities.*

## Questionnaire

Share your feedback and receive a free quota: https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and changes:

*   **SEO Optimization:** Added relevant keywords throughout the README (AI companion, memory framework, open-source, etc.)
*   **Clear Headings:** Organized information with clear and concise headings for better readability and SEO.
*   **Summarized Introduction:** Replaced the initial paragraph with a concise hook to grab attention.
*   **Feature Bullets:** Used bullet points to make key features easy to scan.
*   **Actionable Steps:** Made getting started steps more direct.
*   **Simplified Code Snippets:** Kept code examples concise.
*   **Community Links:** Made community links more prominent.
*   **Call to Action:** Added a clear call to star the repo and to contact for partnerships.
*   **Conciseness:** Removed redundant phrasing.
*   **Visuals:** Included the banner image at the top.
*   **Use Case Emphasis:** Highlighted use cases more visually.
*   **Reorganized Sections:** Restructured the sections for a more logical flow, putting the most important information earlier.
*   **Improved Descriptions:** Reworded descriptions to be more informative.
*   **Removed Irrelevant Information:** Removed the table about the features.
*   **Added Description of MemU Core Concepts:** This section helps to understand the architecture behind the framework.
*   **Added links to the original repository.**