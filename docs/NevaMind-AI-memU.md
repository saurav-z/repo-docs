<div align="center">
  <a href="https://github.com/NevaMind-AI/memU">
    <img src="assets/banner.png" alt="MemU Banner" />
  </a>
</div>

<!-- SEO-optimized introduction with target keywords -->
# MemU: Revolutionizing AI Companion Memory with High Accuracy and Low Cost

**MemU is an open-source memory framework designed to give your AI companions the power of persistent, intelligent memory for engaging and personalized experiences.** This innovative framework offers state-of-the-art accuracy, significant cost reduction, and specialized features tailored for AI companion applications. Explore the future of AI memory with MemU! ([Visit the original repository](https://github.com/NevaMind-AI/memU))

---

## Key Features & Benefits

*   **AI Companion Specialization:** Tailored for optimal performance in AI companion applications.
*   **92% Accuracy:** Achieve state-of-the-art performance, outperforming competitors in the Locomo benchmark.
*   **Up to 90% Cost Reduction:** Optimized online platform delivers significant cost savings.
*   **Advanced Retrieval Strategies:** Utilizing semantic search, hybrid search, and contextual retrieval methods.
*   **24/7 Support:** Dedicated support for enterprise customers.
*   **Autonomous Memory File Management:** Your memories are structured as intelligent folders managed by a memory agent.
*   **Interconnected Knowledge Graph:** Automatically creates meaningful connections between related memories.
*   **Continuous Self-Improvement:** Generates new insights by analyzing existing memories and creates summary documents.
*   **Adaptive Forgetting Mechanism:** Prioritizes information based on usage patterns for a personalized information hierarchy.

---

## Why Choose MemU?

MemU is built with the following advantages:

*   **Higher Memory Accuracy:** Outperforms competitors in the Locomo dataset.
*   **Fast Retrieval:** Optimized retrieval process for efficient access to relevant information.
*   **Low Cost:** Process hundreds of conversation turns to save on costs.

---

## 🚀 Get Started

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The easiest way to integrate MemU:

*   **Instant Access:** Start integrating AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance.

**How to Start:**

1.  **Create an Account:** [https://app.memu.so](https://app.memu.so)
2.  **Generate API Keys:** [https://app.memu.so/api-key/](https://app.memu.so/api-key/)
3.  **Install:** `pip install memu-py`
4.  **Use the example code:**
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
5.  **Explore the Quick Start guide:** [API Reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog).

📖  **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

✨ **That's it!** MemU remembers everything and helps your AI learn from past conversations.

### 🏢 Enterprise Edition

For organizations requiring the most secure and customized experience:

*   **Commercial License:** Full proprietary features, commercial usage rights, white-labeling options.
*   **Custom Development:** SSO/RBAC integration, dedicated algorithm team for scenario-specific framework optimization.
*   **Intelligence & Analytics:** User behavior analysis, real-time production monitoring, automated agent optimization.
*   **Premium Support:** 24/7 dedicated support, custom SLAs, professional implementation services.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers prioritizing control and customization:

*   **Data Privacy:** Keep data within your infrastructure.
*   **Customization:** Modify and extend the platform.
*   **Cost Control:** Avoid recurring cloud fees.

See [self hosting README](README.self_host.md)

---

## ✨ Key Features in Detail

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

### Memory as a File System

*   **Organize** - Autonomous Memory File Management
*   **Link** - Interconnected Knowledge Graph
*   **Evolve** - Continuous Self-Improvement
*   **Never Forget** - Adaptive Forgetting Mechanism

---

## 😺 Advantages & Benefits

*   **Higher Memory Accuracy:** Achieves 92.09% average accuracy in Locomo dataset.  Technical Report will be published soon!

![Memory Accuracy Comparison](assets/benchmark.png)
<em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em>

*   **Fast Retrieval:** Category important information into documents, and only need to find the relevant document content.
*   **Low Cost:** Process hundreds of conversation turns at once.

---

## 🎓 Use Cases:

Explore how MemU can enhance your AI applications:

<div style="display: flex; flex-wrap: wrap; justify-content: center;">
    <div>
        <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>AI Companion
    </div>
    <div>
        <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>AI Role Play
    </div>
    <div>
        <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>AI IP Characters
    </div>
    <div>
        <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>AI Education
    </div>
    <div>
        <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>AI Therapy
    </div>
    <div>
        <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>AI Robot
    </div>
    <div>
        <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>AI Creation
    </div>
     <div>
        More...
    </div>
</div>

---

## 🤝 Contributing

Join the MemU community! Your contributions are welcome:

*   **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Contributions licensed under the **Apache License 2.0**.

---

## 🌍 Community & Support

*   **GitHub Issues:** [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** [Follow us](https://x.com/memU_ai)
*   **Contact:** info@nevamind.ai

---

## 🤝 Ecosystem

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

## 📱 Join Our WeChat Community

Stay connected and get updates:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## Questionnaire

Help us improve! Share your feedback and get free quota: https://forms.gle/H2ZuZVHv72xbqjvd7