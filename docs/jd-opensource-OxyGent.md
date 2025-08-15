# OxyGent: Build Production-Ready Intelligent Systems with Ease

OxyGent is a powerful, open-source framework that empowers developers to rapidly build and deploy multi-agent systems for production environments.  [Explore the OxyGent repository](https://github.com/jd-opensource/OxyGent)

---

## Key Features

*   🚀 **Efficient Multi-Agent Development:** Build, deploy, and evolve AI teams with modular components and clean Python interfaces.
*   🤝 **Intelligent Collaboration:** Leverage dynamic planning for agents to decompose tasks, negotiate solutions, and adapt in real-time.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies and facilitates optimization across distributed systems.
*   🔁 **Continuous Evolution:** Utilize built-in evaluation engines to generate training data and continuously improve agent performance.
*   📈 **Scalability:**  OxyGent's distributed scheduler enables linear cost growth while delivering exponential gains in collaborative intelligence.

OxyGent recently achieved a score of 59.14 on the GAIA benchmark, demonstrating its capabilities in the open-source landscape.

## Software Architecture

### Architecture Diagram

<!-- Insert architecture diagram here -->
### Architecture Overview
*   **Repository:** Centralized storage for agents, tools, LLMs, data, and system files.
*   **Production Framework:** Comprehensive pipeline for registration, building, running, evaluation, and evolution.
*   **Service Framework:** A business system server, offering complete storage and monitoring support.
*   **Engineering Base:** Robust support, including integrated modules like databases and inference engines.

## Feature Highlights: Designed for Developers, Enterprises, and Users

*   **For Developers:** Focus on business logic without reinventing the wheel.
*   **For Enterprises:**  Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

OxyGent provides a complete lifecycle for building intelligent systems:

1.  **Code:** Create agents in Python (no YAML hell)
2.  **Deploy:** Deploy with a single command (local or cloud)
3.  **Monitor:** Track every decision (full transparency)
4.  **Evolve:** Enable automatic improvements (self-improving systems)

## Quick Start

**Prerequisites:** Python 3.10+

**1. Create and activate a Python environment:**

   **Using Conda:**

   ```bash
   conda create -n oxy_env python==3.10
   conda activate oxy_env
   ```

   **Using uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.10 
   uv venv .venv --python 3.10
   source .venv/bin/activate
   ```

**2. Install the OxyGent package:**

   **Using Conda:**

   ```bash
   pip install oxygent
   ```

    **Using uv:**

   ```bash
   uv pip install oxygent
   ```
**3. (Optional) Set up a development environment:**

   1.  Download **[Node.js](https://nodejs.org)**
   2.  Install requirements:

      ```bash
      pip install -r requirements.txt  # or in uv: uv pip install -r requirements.txt
      brew install coreutils # maybe essential
      ```

**4. Write a sample Python script (demo.py):**

```python
   import os
   from oxygent import MAS, Config, oxy, preset_tools

   Config.set_agent_llm_model("default_llm")

   oxy_space = [
      oxy.HttpLLM(
         name="default_llm",
         api_key=os.getenv("DEFAULT_LLM_API_KEY"),
         base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
         model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
         llm_params={"temperature": 0.01},
         semaphore=4,
      ),
      preset_tools.time_tools,
      oxy.ReActAgent(
         name="time_agent",
         desc="A tool that can query the time",
         tools=["time_tools"],
      ),
      preset_tools.file_tools,
      oxy.ReActAgent(
         name="file_agent",
         desc="A tool that can operate the file system",
         tools=["file_tools"],
      ),
      preset_tools.math_tools,
      oxy.ReActAgent(
         name="math_agent",
         desc="A tool that can perform mathematical calculations.",
         tools=["math_tools"],
      ),
      oxy.ReActAgent(
         is_master=True,
         name="master_agent",
         sub_agents=["time_agent", "file_agent", "math_agent"],
      ),
   ]

   async def main():
      async with MAS(oxy_space=oxy_space) as mas:
         await mas.start_web_service(first_query="What time is it now? Please save it into time.txt.")

   if __name__ == "__main__":
      import asyncio
      asyncio.run(main())
```

**5. Configure LLM Settings:**

```bash
   export DEFAULT_LLM_API_KEY="your_api_key"
   export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
   export DEFAULT_LLM_MODEL_NAME="your_model_name"
```

or create a `.env` file:

```
   DEFAULT_LLM_API_KEY="your_api_key"
   DEFAULT_LLM_BASE_URL="your_base_url"
   DEFAULT_LLM_MODEL_NAME="your_model_name"
```

**6. Run the example:**

```bash
python demo.py
```

**7. View the output:**
![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

---

## Contributing

We welcome contributions! You can contribute by:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation:
    *   Fork the repository
    *   Add your views in document
    *   Send your pull request
4.  Writing Code:
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

For development issues, please consult our documentation:  [OxyGent Documentation](http://oxygent.jd.com/docs/)

## Community & Support

For issues or questions, please submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team.

Welcome to contact us:

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

A big thank you to all the [contributors](https://github.com/jd-opensource/OxyGent/graphs/contributors) to OxyGent!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!