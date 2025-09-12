<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Ultimate Memory Framework for AI Companions

**Create AI companions that truly remember with MemU, the open-source memory framework boasting high accuracy, fast retrieval, and cost-effectiveness. [Explore the MemU Repository](https://github.com/NevaMind-AI/memU)**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features

*   **AI Companion Specialization:** Specifically designed to excel in AI companion applications.
*   **Exceptional Accuracy:** Achieve state-of-the-art performance with a 92% accuracy score in the Locomo benchmark.
*   **Significant Cost Savings:** Reduce operational costs by up to 90% through optimized online platform.
*   **Advanced Retrieval Strategies:** Benefit from semantic search, hybrid search, and contextual retrieval for optimal results.
*   **Comprehensive Support:** Enterprise customers receive dedicated 24/7 support.

---

## Why Choose MemU?

MemU empowers you to build AI companions that offer truly personalized experiences:

*   **Remember & Adapt:**  MemU learns about users, their interests, and evolves with every interaction.
*   **Intelligent Memory:**  Manages memories as an intelligent "memory folder," automatically organizing and prioritizing information.
*   **Optimized for AI Companions:** Tailored for the unique needs of AI companion applications, delivering superior performance and user engagement.

---

## Get Started with MemU

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The fastest way to integrate MemU into your application.  Ideal for teams and individuals seeking immediate access without the complexities of setup.

*   **Instant Integration:** Start integrating AI memories in minutes.
*   **Managed Infrastructure:**  We handle scaling, updates, and maintenance.
*   **Premium Support:**  Get priority assistance with a subscription.

**Steps to Get Started:**

1.  **Create an Account:** Register on [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Obtain your API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Integrate into Your Code:**
    ```python
    pip install memu-py

    from memu import MemuClient
    ```
    ```python
    # Initialize
    memu_client = MemuClient(
        base_url="https://api.memu.so", 
        api_key=os.getenv("MEMU_API_KEY")
    )
    memu_client.memorize_conversation(
        conversation=conversation_text, # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
        user_id="user001", 
        user_name="User", 
        agent_id="assistant001", 
        agent_name="Assistant"
    )
    ```
    **That's it!** MemU remembers everything and helps your AI learn from past conversations.

    *For complete integration details, see [`example/client/memory.py`](example/client/memory.py)*

### 🏢 Enterprise Edition

For organizations demanding the highest levels of security, customization, and control.

*   **Commercial Licensing:**  Access full proprietary features and commercial usage rights.
*   **Custom Development:**  Benefit from SSO/RBAC integration and a dedicated algorithm team.
*   **Advanced Intelligence & Analytics:**  Utilize user behavior analysis, real-time monitoring, and agent optimization.
*   **Premium Support:**  Receive 24/7 dedicated support, custom SLAs, and implementation services.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers prioritizing local control, data privacy, and customization.

*   **Data Privacy:** Maintain complete control over your sensitive data.
*   **Customization:** Tailor the platform to perfectly suit your needs.
*   **Cost Control:** Avoid recurring cloud fees with large-scale deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Key Advantages & Benefits

### Higher Memory Accuracy

MemU achieves a 92.09% average accuracy on the Locomo dataset, significantly surpassing the competition. (Technical Report coming soon!)

![Memory Accuracy Comparison](assets/benchmark.png)
*(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts;*

### Fast Retrieval

Efficiently retrieve information by categorizing important data into documents. This eliminates the need for extensive embedding searches for every sentence, resulting in faster retrieval times.

### Low Cost

Process hundreds of conversation turns simultaneously, reducing the need for repeated memory function calls. This strategy helps save tokens and reduces operational costs.  See [best practice](https://memu.pro/blog/memu-best-practice).

---

## 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

## Memory as File System

### **Organize** - Autonomous Memory File Management

*   Your memories are structured as intelligent folders managed by a memory agent. The agent automatically decides what to record, modify, or archive, functioning like a personal librarian.

### **Link** - Interconnected Knowledge Graph

*   Memories automatically connect to related information, building a rich network of hyperlinked documents.

### **Evolve** - Continuous Self-Improvement

*   Your memory agent generates new insights by analyzing existing memories, identifying patterns, and creating summary documents.

### **Never Forget** - Adaptive Forgetting Mechanism

*   The agent prioritizes information based on usage patterns, ensuring the most relevant content remains readily accessible.

---

## 🎓 Use Cases

|                       |                       |                       |                       |
| :-------------------: | :-------------------: | :-------------------: | :-------------------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More... |

---

## 🤝 Contribute to MemU

We build trust through open-source collaboration.  Your contributions are vital to MemU's continued innovation!

*   Explore our GitHub issues and projects.
*   Make your mark on the future of AI memory!

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Your contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support, chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and announcements. [Follow us](https://x.com/memU_ai)
*   **Email:** For more information please contact info@nevamind.ai

---

## 🤝 Ecosystem

We're proud to work with these amazing organizations:

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

## 📱 Join Our WeChat Community

Connect with us on WeChat for the latest updates, community discussions, and exclusive content:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">

*Scan the QR code above to join our WeChat community*

</div>

---

*Stay connected with the MemU community! Join our WeChat groups for real-time discussions, technical support, and networking opportunities.*

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7