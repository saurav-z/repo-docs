<div align="center">
  <a href="https://github.com/NevaMind-AI/memU">
    <img src="assets/banner.png" alt="MemU Banner" width="100%">
  </a>
</div>

# MemU: Unlock Unforgettable AI Companions with Next-Gen Memory

**MemU is an open-source memory framework designed to empower AI companions with exceptional recall, speed, and cost-efficiency, enabling them to truly remember and learn from every interaction.**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)
[![Reddit](https://img.shields.io/badge/Reddit-Join%20Community-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/r/memU_ai)
[![WeChat](https://img.shields.io/badge/WeChat-WeChat%20Group-07C160?logo=wechat&logoColor=white)](assets/wechat/wechat1.png)

*   [**Visit our homepage: memu.pro**](https://memu.pro/)

## Key Features & Benefits

*   **AI Companion Focused:** Specifically designed for AI companion applications, optimizing for their unique memory needs.
*   **Unmatched Accuracy (92%):** Achieves a state-of-the-art 92% accuracy score on the Locomo benchmark, surpassing competitors.
*   **Significant Cost Reduction (Up to 90%):** Optimized online platform for cost-effective memory management.
*   **Advanced Retrieval Strategies:** Employs multiple retrieval methods, including semantic, hybrid, and contextual search.
*   **24/7 Enterprise Support:** Dedicated support for enterprise customers to ensure seamless integration and optimal performance.

---

## Why Choose MemU?

MemU empowers you to build AI companions that:

*   **Remember Personal Details:** Learn and recall user preferences, conversation history, and more.
*   **Evolve Through Interactions:** Grow and adapt with each interaction, becoming increasingly personalized.
*   **Provide Engaging Experiences:** Create AI companions that feel intelligent and responsive.

---

## Get Started: Quick & Easy Integration

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The fastest way to integrate your application with memU. Perfect for teams and individuals who want immediate access without setup complexity. We host the models, APIs, and cloud storage, ensuring your application gets the best quality AI memory.

*   **Instant Access** - Start integrating AI memories in minutes
*   **Managed Infrastructure** - We handle scaling, updates, and maintenance for optimal memory quality
*   **Premium Support** - Subscribe and get priority assistance from our engineering team

**Step 1: Create an Account**
Create an account on [https://app.memu.so](https://app.memu.so).

**Step 2: Generate API Keys**
Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) to generate your API keys.

**Step 3: Integrate into your code**
```python
pip install memu-py

# Example usage
from memu import MemuClient
```

**Step 4: Quick Start**
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
**See `example/client/memory.py` for complete integration details**

📖 **API Reference:**  [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

✨ **That's it!** MemU remembers everything and helps your AI learn from past conversations.

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, control and best quality:

*   **Commercial License** - Full proprietary features, commercial usage rights, white-labeling options
*   **Custom Development** - SSO/RBAC integration, dedicated algorithm team for scenario-specific framework optimization
*   **Intelligence & Analytics** - User behavior analysis, real-time production monitoring, automated agent optimization
*   **Premium Support** - 24/7 dedicated support, custom SLAs, professional implementation services

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers who prefer local control, data privacy, or customization:

*   **Data Privacy** - Keep sensitive data within your infrastructure
*   **Customization** - Modify and extend the platform to fit your needs
*   **Cost Control** - Avoid recurring cloud fees for large-scale deployments

See [self hosting README](README.self_host.md)

---

## Deep Dive: MemU's Core Technologies

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### Memory as a File System: The Intelligent Memory Agent

*   **Organize:**  Memories are organized as intelligent folders, autonomously managed by the memory agent.
*   **Link:**  The system creates connections between related memories, building a knowledge graph.
*   **Evolve:**  The memory agent generates insights, identifies patterns, and creates summaries.
*   **Never Forget:** Adaptive forgetting prioritizes information based on usage, creating a personalized information hierarchy.

---

## Advantages of MemU

*   **Superior Accuracy:** Achieves 92.09% average accuracy in Locomo across all reasoning tasks.
    ![Memory Accuracy Comparison](assets/benchmark.png)
*   **Blazing Fast Retrieval:** Efficiently retrieves relevant information by categorizing and indexing important data.
*   **Cost-Effective:** Processes conversations efficiently, optimizing token usage and reducing overall costs. See [best practice](https://memu.pro/blog/memu-best-practice).

---

## Explore MemU: Use Cases

| Use Case          | Description                                                    |
| :---------------- | :------------------------------------------------------------- |
| AI Companion      | Create intelligent AI companions that remember users.          |
| AI Role Play      | Enhance AI role-playing with persistent memory.                |
| AI IP Characters  | Develop AI characters with detailed backstories and knowledge. |
| AI Education      | Build personalized learning experiences with AI tutors.       |
| AI Therapy        | Provide supportive and consistent AI therapy.                  |
| AI Robot          | Enable robots to learn and adapt through memory.             |
| AI Creation       | Support AI-driven content creation with retained knowledge.    |

<div align="center">
  <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200">
  <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200">
  <img src="assets/usecase/ai_ip-0000.png" width="150" height="200">
  <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200">
  <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200">
  <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200">
  <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200">
  <!-- Add more use case images here -->
</div>

---

## Contribute to MemU

We encourage open-source collaboration. Contribute and help shape the future of MemU.

*   **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### License

Contributions are licensed under the **Apache License 2.0**.

---

## Join Our Community

*   **GitHub Issues:** Report bugs and request features: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and connect: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated: [Follow us](https://x.com/memU_ai)
*   **Contact us:** [info@nevamind.ai](mailto:info@nevamind.ai)

---

## Ecosystem & Partners

We're proud to work with these amazing organizations:

<div align="center">
  <h3>Development Tools</h3>
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

*Interested in partnering with MemU?  [Contact us](mailto:contact@nevamind.ai)*

---

## Stay Connected: Join Our WeChat Community

<div align="center">
  <img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
  <br>
  <em>Scan the QR code to join the MemU WeChat community.</em>
</div>

---

*  [**Back to top**](#memu-unlock-unforgettable-ai-companions-with-next-gen-memory)
*   **[View the original repository on GitHub](https://github.com/NevaMind-AI/memU)**
```

Key improvements and SEO optimizations:

*   **Concise, compelling introduction:**  The hook immediately grabs attention.
*   **Clear Heading Structure:**  Uses H2 and H3 headings for improved readability and SEO.
*   **Keyword Optimization:**  Includes relevant keywords like "AI companion," "memory framework," "open source," "AI," and  "memory" throughout the text.
*   **Bulleted Lists:**  Uses bullet points for key features, benefits, and steps to get started.
*   **Strong Call to Action:** Encourages users to "Get Started," "Join Our Community," and "Contribute."
*   **Simplified "Get Started" Section:**  Makes the setup process as easy as possible, highlighting the cloud option.
*   **Emphasis on Benefits:** Focuses on what users *get* (e.g., "Unlock Unforgettable AI Companions").
*   **Use Case Examples:** Showcases the versatility of MemU with concise and eye-catching examples.
*   **Community Engagement:**  Highlights ways to connect and contribute.
*   **Clear Links:** Provides direct links to relevant resources, including the API Reference.
*   **Removed redundant information:** streamlined the text and avoided repetition.
*   **Partner section kept and updated:** Highlights existing integrations.
*   **Clear organization and formatting:** The structure is improved for clarity and readability.
*   **Link back to the original repo added.**