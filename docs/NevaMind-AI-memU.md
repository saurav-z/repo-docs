<div align="center">

![MemU Banner](assets/banner.png)
</div>

# MemU: Revolutionizing AI Companions with Advanced Memory

**MemU is an open-source memory framework that empowers your AI companions to learn, remember, and grow with unparalleled accuracy and efficiency.** ([View the original repo on GitHub](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

## Key Features of MemU

*   **AI Companion Specialization:** Tailored specifically for AI companion applications.
*   **High Accuracy:** Achieve state-of-the-art performance with 92% accuracy in Locomo benchmark tests.
*   **Cost-Effective:** Reduce costs by up to 90% through optimized online platform usage.
*   **Advanced Retrieval Strategies:** Utilize multiple methods, including semantic search, hybrid search, and contextual retrieval for superior recall.
*   **Enterprise-Grade Support:** Access 24/7 support for enterprise customers.

---

## Why Choose MemU?

MemU goes beyond simple memory storage; it's an intelligent "memory folder" that adapts to your AI companion's needs. It helps your AI companions learn who you are, understand your preferences, and evolve alongside you through every interaction.

*   **Unforgettable Conversations:** MemU ensures your AI companions remember and leverage past interactions.
*   **Effortless Integration:** Quickly integrate MemU into your projects with our cloud and self-hosting options.
*   **Community Driven:** Join a vibrant community of AI developers and contribute to the future of AI.

## Get Started with MemU

Choose the best method for your needs:

### ☁️ **Cloud Version (Online Platform)**

The fastest way to integrate with memU. Ideal for teams and individuals seeking immediate access without setup complexities.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance from our engineering team.

**Steps:**

1.  Create an account at [https://app.memu.so](https://app.memu.so).
2.  Generate API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  Integrate with these 3 lines of code.

```python
pip install memu-py

# Example usage
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

Check out the [API reference](docs/API_REFERENCE.md) or the [blog](https://memu.pro/blog) for details.

📖 **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

✨ **That's it!** MemU remembers everything and helps your AI learn from past conversations.

### 🏢 **Enterprise Edition**

For organizations seeking maximum security, customization, control, and optimal quality:

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** Integration options and dedicated algorithm team support.
*   **Intelligence & Analytics:** User behavior analysis and production monitoring.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 **Self-Hosting (Community Edition)**

For users and developers who prefer local control, data privacy, and customization:

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform.
*   **Cost Control:** Avoid cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Unveiling MemU's Core Concepts

### 🎥 **Demo Video**

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### 🧠 **Memory as File System**

#### Organize: Autonomous Memory File Management

Your memories are structured as intelligent folders managed by a memory agent. The memory agent automatically decides what to record, modify, or archive, acting like your personal librarian.

#### Link: Interconnected Knowledge Graph

Memories are interconnected. The system automatically creates relationships between memories, transforming search into effortless recall.

#### Evolve: Continuous Self-Improvement

Even offline, your memory agent works. It generates new insights by analyzing memories and creating summaries, making your knowledge base smarter over time.

#### Never Forget: Adaptive Forgetting Mechanism

The memory agent prioritizes information based on usage. Recently accessed memories stay accessible, while less relevant content is deprioritized.

---

## 🌟 Advantages of MemU

### Higher Memory Accuracy

MemU achieves 92.09% average accuracy on the Locomo dataset.

![Memory Accuracy Comparison](assets/benchmark.png)
_(1) Single-hop questions; (2) Multi-hop questions; (3) Temporal reasoning questions; (4) Open-domain knowledge questions)_

### Fast Retrieval

MemU categorizes important information into documents. During retrieval, we only need to find the relevant document content, eliminating the need for extensive embedding searches.

### Low Cost

MemU processes hundreds of conversation turns at once, saving tokens and money. See [best practice](https://memu.pro/blog/memu-best-practice).

---

## 🚀 Use Cases

MemU is versatile, supporting a range of AI applications:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contribute to MemU

We welcome your contributions!

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### 📄 License

Your contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:** [Submit an issue](https://github.com/NevaMind-AI/memU/issues) for bug reports, feature requests, and project tracking.
*   **Discord:** [Join us](https://discord.com/invite/hQZntfGsbJ) for real-time support, community chat, and updates.
*   **X (Twitter):** [Follow us](https://x.com/memU_ai) for the latest news.

---

## 🤝 Ecosystem

MemU works with great organizations:

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

*Partner with MemU? [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## 📱 Join Our WeChat Community

Stay connected! Scan the QR code to join our WeChat community for the latest updates and discussions:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

*Scan the QR codes above to join our WeChat community*

---

## 📝 Help Us Improve

Share your feedback in our 3-minute survey and get free quota! [https://forms.gle/H2ZuZVHv72xbqjvd7](https://forms.gle/H2ZuZVHv72xbqjvd7)