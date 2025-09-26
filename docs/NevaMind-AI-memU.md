<div align="center">
  <img src="assets/banner.png" alt="MemU Banner" width="100%">
</div>

# MemU: The Cutting-Edge Memory Framework for Intelligent AI Companions

**Build AI companions that truly remember and grow with MemU, the open-source memory framework designed for high accuracy, speed, and cost-effectiveness.**  ([Original Repo](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features of MemU: Revolutionizing AI Companion Memory

*   ✅ **AI Companion Specialization:** Designed specifically for AI companion applications.
*   ✅ **Unmatched Accuracy:** Achieve state-of-the-art performance with 92% accuracy in the Locomo benchmark.
*   ✅ **Cost-Effective Solutions:** Reduce costs by up to 90% through optimized online platform and efficient architecture.
*   ✅ **Advanced Retrieval Strategies:** Leverage semantic search, hybrid search, and contextual retrieval for optimal results.
*   ✅ **24/7 Enterprise Support:** Dedicated support for enterprise customers.

---

## Why Choose MemU? Advantages & Benefits

*   **Higher Memory Accuracy:** MemU leads the pack with an impressive 92.09% average accuracy in the Locomo dataset across all reasoning tasks, outperforming competitors.
*   **Fast Retrieval:** Efficiently retrieve the most relevant information by categorizing important data into documents, eliminating the need for extensive embedding searches.
*   **Low Cost:** Process hundreds of conversation turns at once, saving on token usage and memory operation costs.

---

## Get Started with MemU

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

Integrate AI memories in minutes with our cloud version, ideal for teams and individuals.

*   **Instant Access:** Begin integrating AI memories immediately.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Receive priority assistance.

**Quick Start Guide:**

1.  **Create an Account:** Sign up at [https://app.memu.so](https://app.memu.so)
2.  **Generate API Keys:** Obtain API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/)
3.  **Install the Library:** `pip install memu-py`
4.  **Example Usage:**

```python
from memu import MemuClient
import os

# Initialize
memu_client = MemuClient(
    base_url="https://api.memu.so",
    api_key=os.getenv("MEMU_API_KEY")
)
conversation_text = "Your conversation text here..." # Replace with actual conversation
memu_client.memorize_conversation(
    conversation=conversation_text, # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
    user_id="user001",
    user_name="User",
    agent_id="assistant001",
    agent_name="Assistant"
)
```

**See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

### 🏢 Enterprise Edition

For organizations needing maximum security, customization, and control:

*   **Commercial License:** Proprietary features and commercial usage rights.
*   **Custom Development:** SSO/RBAC integration and tailored algorithm optimization.
*   **Intelligence & Analytics:** User behavior analysis and agent optimization.
*   **Premium Support:** Dedicated 24/7 support and custom SLAs.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

Control your data and customize to your needs:

*   **Data Privacy:** Keep your data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Dive Deeper: How MemU Works

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### Memory as file system

#### **Organize** - Autonomous Memory File Management
Memories are intelligently structured as folders managed by a memory agent.

#### **Link** - Interconnected Knowledge Graph
The system creates connections between memories.

#### **Evolve** - Continuous Self-Improvement
Your memory agent generates new insights by analyzing existing memories.

#### **Never Forget** - Adaptive Forgetting Mechanism
The memory agent prioritizes information based on usage patterns.

---

## 🎓 Explore MemU Use Cases

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contribute to MemU

We welcome your contributions! Join us in building the future of AI memory.

*   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**
*   **License:** Your contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Stay Connected with the MemU Community

*   **GitHub Issues:** Report bugs, suggest features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated. [Follow us](https://x.com/memU_ai)
*   **Email:** For more information please contact [info@nevamind.ai](mailto:info@nevamind.ai)

---

## 🤝 Ecosystem of Partners

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

## 📱 Stay Updated: Join our WeChat Community

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and SEO considerations:

*   **Concise Hook:**  A strong opening sentence to grab attention.
*   **Targeted Keywords:** Incorporated keywords like "AI companion," "memory framework," "open-source," and "AI companion applications."
*   **Clear Headings:**  Used descriptive headings to improve readability and organization (e.g., "Key Features," "Why Choose MemU?," "Get Started," "Explore Use Cases").
*   **Bulleted Lists:**  Emphasized key features and benefits with bulleted lists for easy scanning.
*   **Stronger Call to Action:**  Encouraged users to "Get Started," "Join the Community," and "Contribute."
*   **SEO Optimization:** Used headings, and keywords and optimized text.
*   **Improved Structure:** Reorganized the content for better flow and readability.
*   **Concise Language:** Removed unnecessary words and phrases.
*   **Focus on Benefits:** Highlighted what users gain from using MemU.
*   **Clear Examples:** Made the Quick Start guide clearer.
*   **Complete Links:** Ensured all links are working.