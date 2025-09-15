## Automate Your Telegram Tasks with tg-signer

Tired of manually interacting with Telegram? **tg-signer** is your all-in-one solution for automating daily check-ins, message monitoring, and intelligent responses.  [Check out the original repository](https://github.com/amchii/tg-signer) for the full details!

### Key Features

*   **Automated Check-ins:** Schedule and execute daily tasks with customizable timings and random delays.
*   **Intelligent Interactions:** Automate actions like clicking keyboard buttons based on configured text.
*   **AI-Powered Automation:** Utilize AI to recognize images and interact with Telegram's interface.
*   **Personalized Monitoring:**  Monitor and automatically respond to messages in your personal chats, groups, or channels.
*   **Flexible Automation Flows:** Design custom action sequences for complex Telegram interactions.
*   **Multi-Account Support:**  Run multiple Telegram accounts with a single configuration.
*   **Message Scheduling:** Leverage Telegram's built-in scheduling for timed message delivery.
*   **Message Forwarding:** Forward messages to external services via UDP or HTTP.

### Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For optimized performance:

```bash
pip install "tg-signer[speedup]"
```

#### Docker

Build your own Docker image using the provided [Dockerfile](./docker) and the related [README](./docker/README.md).

### Usage

```bash
tg-signer run
tg-signer run my_sign  # Run a task immediately, skipping prompts
tg-signer run-once my_sign  # Execute a task once, regardless of the schedule
tg-signer send-text 8671234001 /test  # Send a message to a specific chat
tg-signer monitor run my_monitor # Configure and start monitoring
```

For detailed command options and examples, use the `--help` flag with any subcommand.

### Configuration

*   **Proxy:** Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line argument.
*   **Login:** Use `tg-signer login` to authenticate your Telegram account and fetch your recent chats.

### Data Storage

Configuration and data are stored in the `.signer` directory by default.