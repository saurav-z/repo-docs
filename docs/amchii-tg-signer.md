# tg-signer: Automate Telegram Tasks - Sign-ins, Monitoring, and More!

**Tired of manual Telegram interactions? Automate your daily sign-ins, monitor chats, and create custom responses with tg-signer!**  [View the original repository](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Sign-ins:**  Schedule and automate daily sign-ins with customizable time variations.
*   **Intelligent Keyboard Interaction:**  Click buttons based on configured text or use AI-powered image recognition.
*   **Advanced Monitoring:**  Monitor personal chats, groups, and channels, with options for message forwarding and automated replies.
*   **Flexible Action Flows:**  Define custom action sequences for various tasks.
*   **AI Integration:** Use AI to recognize images and solve math problems.
*   **Multi-Account Support:** Run multiple accounts concurrently with the `multi-run` command.
*   **Configuration Management:** Easily import/export configurations and monitor/sign-in logs.
*   **Message Scheduling:** Schedule messages using Telegram's built-in functionality.

## Installation

Requires Python 3.9 or higher.

**Install the base package:**

```bash
pip install -U tg-signer
```

**For faster performance (optional):**

```bash
pip install "tg-signer[speedup]"
```

**Docker Support:**

Build your own Docker image using the provided [Dockerfile](./docker/Dockerfile) and associated [README](./docker/README.md) in the `docker` directory.

## Usage

```
tg-signer [OPTIONS] COMMAND [ARGS]...
```

Use `<subcommand> --help` for detailed instructions on each command.

**Key Subcommands:**

*   `run`: Execute sign-in tasks based on configuration.
*   `run-once`: Run a sign-in task once, regardless of previous execution.
*   `send-text`: Send a text message to a specific chat.
*   `monitor`: Configure and run chat monitoring with automated responses.
*   `multi-run`: Run a task across multiple Telegram accounts simultaneously.
*   `login`: Log in to your Telegram account.
*   `logout`: Log out of your Telegram account and delete the session file.

**Example Commands:**

```bash
tg-signer run
tg-signer run my_sign  # Run 'my_sign' task
tg-signer run-once my_sign  # Run 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat ID 8671234001
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer monitor run # Configure and run monitoring
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' for multiple accounts
```

## Proxy Configuration

`tg-signer` does not read system proxies. Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option.

**Example:**

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

## Logging In

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in and retrieve your chat list.

## Configuration and Data Storage

Configurations and data are stored in the `.signer` directory.  Use `tree .signer` to view the file structure:

```
.signer
├── latest_chats.json  # Recent chats
├── me.json  # Personal information
├── monitors          # Monitoring configurations
│   ├── my_monitor  # Monitoring task name
│       └── config.json  # Monitoring configuration
└── signs           # Sign-in task configurations
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in records
```

## Version History

**(See original README for specific version change logs.)**