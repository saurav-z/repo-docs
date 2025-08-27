<!-- SEO-optimized README for MemU -->
<div align="center">

![MemU Banner](assets/banner.png)

### MemU: The Next-Gen Memory Framework for AI Companions

</div>

**Unlock the power of persistent memory for your AI companions with MemU, the open-source framework for building AI that truly remembers.**  MemU empowers you to create AI companions that learn, adapt, and grow with every interaction.  

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

[**➡️ Visit the MemU GitHub Repository for detailed information and code!**](https://github.com/NevaMind-AI/memU)

## Key Features of MemU:

*   ✅ **AI Companion Specialization:** Optimized for AI companion applications.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance in the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimize costs through our online platform.
*   ✅ **Advanced Retrieval Strategies:**  Employ multiple methods, including semantic search, hybrid search, and contextual retrieval, for accurate information retrieval.
*   ✅ **24/7 Support:** Available for enterprise customers.

---

## Why Choose MemU?

MemU revolutionizes how AI companions remember and learn.  We offer:

*   **Unmatched Accuracy:** Benefit from our leading accuracy in memory recall.
*   **Significant Cost Savings:** Reduce your operational costs with MemU's optimized infrastructure.
*   **Fast & Efficient Retrieval:** Quickly access the information you need.

---

## 🚀 Get Started

Choose the best option for your needs:

### ☁️ **Cloud Version (Online Platform: [app.memu.so](https://app.memu.so))**

The fastest way to integrate AI memory into your application.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance.

**Steps:**

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:**  Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) to generate API keys.
3.  **Integrate in Code:**
    ```python
    pip install memu-py

    # Example usage
    from memu import MemuClient
    import os

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

    **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

### 🏢 Enterprise Edition

For organizations requiring security, customization, and control:

*   **Commercial License:** Full proprietary features and white-labeling.
*   **Custom Development:** SSO/RBAC, dedicated algorithm optimization.
*   **Intelligence & Analytics:** User behavior analysis, real-time monitoring, and automated agent optimization.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For local control, data privacy, and customization:

*   **Data Privacy:** Keep your data within your infrastructure.
*   **Customization:** Modify and extend the platform.
*   **Cost Control:** Avoid recurring cloud fees.

See [self hosting README](README.self_host.md)

---

## ✨ Core MemU Concepts

### 🎥 **Demo Video**

Watch the MemU demonstration video:

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### Memory as File System

*   **Organize:**  Autonomous Memory File Management, organizing memories intelligently.
*   **Link:** Interconnected Knowledge Graph, building connections between related memories.
*   **Evolve:** Continuous Self-Improvement, generating new insights and patterns over time.
*   **Never Forget:**  Adaptive Forgetting Mechanism, prioritizing relevant information based on usage.

---

## 😺 Advantages in Detail

### Higher Memory Accuracy

MemU achieves 92.09% average accuracy in the Locomo dataset across all reasoning tasks, outperforming competitors.
![Memory Accuracy Comparison](assets/benchmark.png)

### Fast Retrieval

Important information is categorized into documents, and during retrieval, we only need to find the relevant document content, eliminating the need for extensive embedding searches for fragmented sentences.

### Low Cost

Process hundreds of conversation turns at once, eliminating the need for developers to repeatedly call memory functions, thus saving users from wasting tokens on multiple memory operations.

---

## 🎓 Use Cases

MemU can power a variety of AI applications:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contributing

We welcome your contributions!  Help us shape the future of MemU.

📋 **[Read our detailed Contributing Guide](CONTRIBUTING.md)**

### **📄 License**

Your contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community & Support

*   **GitHub Issues:** Report bugs, request features, and track development: [Submit an issue](https://github.com/NevaMind-AI/memU/issues).
*   **Discord:** Join our community for real-time support: [Join our Discord](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates: [Follow us on Twitter](https://x.com/memU_ai)
*   **General Inquiries:** info@nevamind.ai

---

## 🤝 Ecosystem Partners

We are proud to collaborate with these leading organizations:

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

Stay connected with the MemU community through our WeChat groups:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## 🙋‍♀️ Questionnaire

Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and explanations:

*   **SEO Optimization:**  Included relevant keywords like "AI companion," "memory framework," and "AI."  The title and headings are clear and descriptive.
*   **One-Sentence Hook:**  A compelling opening sentence captures the essence of MemU.
*   **Clear Structure with Headings:**  The README is well-organized for easy navigation.  Uses appropriate H2 and H3 headings.
*   **Bulleted Key Features:** Highlights the main advantages and benefits.
*   **Concise Language:**  Avoids overly verbose descriptions.
*   **Call to Actions (CTAs):**  Encourages users to explore the GitHub repo, join the community, and contribute.
*   **Emphasis on Benefits:**  Focuses on *what* users can achieve with MemU (e.g., "Unlock the power...").
*   **Clear "Getting Started" Section:**  Provides a straightforward guide to different integration methods.
*   **Complete Code Example:** The code example is included, so users don't have to copy-paste it from the original.
*   **Community Building:**  Highlights the active community and various ways to connect.
*   **Partners Section:**  Showcases supporting partners and tools.
*   **Contact information:** Clear info on how to reach the team and the community.
*   **Removed redundant text:** Removed repetitive phrases.
*   **WeChat Information:** Kept this since it is unique to the documentation.
*   **Survey Encouragement:** Kept this as it's an invitation for feedback.

This revised README is much more user-friendly, informative, and optimized for both human readers and search engines.  It effectively communicates the value proposition of MemU and encourages engagement.