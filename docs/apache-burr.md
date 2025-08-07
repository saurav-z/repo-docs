# Burr: Build, Monitor, and Debug State-Driven AI Applications

Burr is a powerful Python library that simplifies building stateful AI applications like chatbots and agents by enabling you to express your application as a state machine.  [Explore the Burr GitHub repository](https://github.com/apache/burr).

**Key Features:**

*   **State Machine Framework:**  Define your application's logic using a clear, state machine model.
*   **Real-time UI:**  Visualize and debug your application's execution with a built-in telemetry UI.
*   **Flexible Integrations:**  Easily integrate with LLMs, storage solutions, and other frameworks.
*   **Idempotent Workflows:**  Manage state, handle complex decisions, and create self-persisting workflows.
*   **Framework Agnostic:** Integrate with your favorite frameworks, libraries, and vendors.

## Get Started Quickly

1.  **Install:** `pip install "burr[start]"`
2.  **Run the UI:** `burr`
3.  **Explore Demos:**  Navigate to the "Demos" section in the UI to see example applications, including a chatbot.
4.  **Run an example:**  Clone the repository and run a hello world example: `git clone https://github.com/apache/burr && cd burr/examples/hello-world-counter && python application.py`

Refer to the [documentation](https://burr.dagworks.io/) and [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for detailed instructions.

## How Burr Works

Burr uses a simple API to define state machines, making it easy to manage and track your AI application's state.  Key components include:

*   **Actions:**  Python functions that perform operations and potentially modify the application's state.
*   **State:**  Represents the current condition of your application.
*   **Transitions:**  Define the flow between states.
*   **UI:** A UI to monitor your applications in real-time and debug any issues.

## Applications of Burr

Burr can power a variety of applications, including:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Interactive assistants for writing emails
*   Simulations
*   Hyperparameter tuning

## Comparison with Other Frameworks

Burr offers a unique approach to building AI applications.

| Criteria                                          | Burr | Langgraph | Temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
>
> **Ashish Ghosh**, *CTO, Peanut Robotics*

> "Moving from LangChain to Burr was a game-changer!"
>
> **Aditya K.**, *DS Architect, TaskHuman*

## Roadmap

Future plans include:

*   FastAPI integration and hosted deployment
*   Core library efficiency and usability improvements (retries, exception handling, metadata)
*   Framework integrations (LCEL, LlamaIndex, Hamilton)
*   Hosted execution of state machines
*   Additional storage integrations

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) for instructions on how to get involved.

## Contributors

Thank you to all contributors!

*   [Elijah ben Izzy](https://github.com/elijahbenizzy)
*   [Stefan Krawczyk](https://github.com/skrawcz)
*   [Joseph Booth](https://github.com/jombooth)
*   [Nandani Thakur](https://github.com/NandaniThakur)
*   [Thierry Jean](https://github.com/zilto)
*   [Hamza Farhan](https://github.com/HamzaFarhan)
*   [Abdul Rafay](https://github.com/proftorch)
*   [Margaret Lange](https://github.com/margaretlange)
*   And more!