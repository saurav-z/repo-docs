# Burr: Build and Monitor Stateful AI Applications

**Burr empowers developers to easily build stateful AI applications like chatbots and agents using simple Python building blocks and a powerful UI.**

[![Discord](https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord)](https://discord.gg/6Zy2DwP4f3)
[![Downloads](https://static.pepy.tech/badge/burr/month)](https://pepy.tech/project/burr)
![PyPI Downloads](https://static.pepy.tech/badge/burr)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/dagworks-inc/burr)](https://github.com/dagworks-inc/burr/pulse)
[![X](https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social)](https://twitter.com/burr_framework)
<a target="_blank" href="https://linkedin.com/showcase/dagworks-inc" style="background:none">
  <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" />
</a>
<a href="https://twitter.com/burr_framework" target="_blank">
  <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X"/>
</a>
<a href="https://twitter.com/dagworks" target="_blank">
  <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X"/>
</a>

**Key Features:**

*   **Simple Python API:** Build state machines with intuitive Python functions.
*   **Real-time UI:** Monitor, trace, and debug your applications with a built-in UI.
*   **Pluggable Persistence:** Save and load application state with ease.
*   **Framework Agnostic:** Integrates seamlessly with your favorite LLM frameworks.
*   **Comprehensive Examples:** Start building with various example applications, including a simple gpt-like chatbot and a stateful RAG-based chatbot.

**Get Started Quickly:**

1.  **Installation:** `pip install "burr[start]"`
2.  **Run the UI:** `burr`
3.  **Explore Examples:** See how the counter example and demo chatbot work in the UI!

**Explore the Examples:**

*   [Simple gpt-like chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
*   [Stateful RAG-based chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
*   [LLM-based adventure game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
*   [Interactive assistant for writing emails](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)
*   And more!

**How Burr Works:**

Burr lets you model applications as state machines. Your application's logic can be broken down into steps and transitions between these states. This lets you manage complex decisions, handle user feedback, and build idempotent, self-persisting workflows.

**Learn More:**
*   [Documentation](https://burr.dagworks.io/)
*   [Quick Intro Video](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9)
*   [Detailed Video Walkthrough](https://www.youtube.com/watch?v=rEZ4oDN0GdU)
*   [Blog Post](https://blog.dagworks.io/p/burr-develop-stateful-ai-applications)
*   [Join the Discord](https://discord.gg/6Zy2DwP4f3)

**Why Burr?**

Burr allows you to build stateful LLM apps in a reliable and testable manner. It allows for better debugging and faster experimentation. See the [Comparison Against Common Frameworks](#comparison-against-common-frameworks) to see what Burr does well compared to the competition!

**Comparison Against Common Frameworks**

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

**Testimonials**

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."

**Ashish Ghosh**
*CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."

**Reddit user cyan2k**
*LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."

**Ishita**
*Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."

**Matthew Rideout**
*Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."

**Rinat Gareev**
*Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."

**Hadi Nayebi**
*Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."

**Aditya K.**
*DS Architect, TaskHuman*

**Roadmap**

*   FastAPI Integration + Hosted Deployment
*   Core Library Efficiency and Usability Improvements
*   Tooling for Hosted Execution
*   Additional Storage Integrations

Sign up for the Burr Cloud waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

**Contributing**

We welcome contributions!  Check out the [developer docs](https://burr.dagworks.io/contributing) to get started.

**[Original Repository](https://github.com/apache/burr)**