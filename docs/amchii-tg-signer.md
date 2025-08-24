# tg-signer: Automate Telegram Tasks with Python

**Effortlessly automate your Telegram interactions with `tg-signer`, a versatile Python-based tool for daily check-ins, message monitoring, and automated responses.**  [View the original repository on GitHub](https://github.com/amchii/tg-signer)

## Key Features

*   **Automated Check-ins:**  Schedule daily check-ins with customizable timing and error margins.
*   **Keyboard Interaction:**  Automate interactions by clicking on keyboard buttons based on configured text or image recognition.
*   **AI-Powered Image Recognition:**  Leverage AI to identify images and click relevant keyboard options.
*   **Message Monitoring & Automation:**  Monitor, forward, and automatically reply to messages in personal chats, groups, and channels.
*   **Customizable Action Flows:**  Configure and execute complex action sequences.
*   **Multi-Account Support:** Run tasks with multiple accounts simultaneously.
*   **Message Scheduling:**  Configure Telegram's built-in message scheduling features.

## Installation

Ensure you have Python 3.9 or higher installed.

Install using pip:

```bash
pip install -U tg-signer
```

For improved speed, install with the speedup option:

```bash
pip install "tg-signer[speedup]"
```

## Docker

While a pre-built Docker image isn't provided, you can build your own using the `Dockerfile` and accompanying `README` located in the `./docker` directory.

## Usage

The `tg-signer` command-line tool provides several subcommands:

```
tg-signer [OPTIONS] COMMAND [ARGS]...
```

Use `<subcommand> --help` for detailed usage instructions for each command.

**Example Commands:**

*   `tg-signer run`:  Run a configured check-in task.
*   `tg-signer run my_sign`: Run the check-in task named 'my_sign'.
*   `tg-signer run-once my_sign`: Run the check-in task 'my_sign' once, even if it has already been run today.
*   `tg-signer send-text 8671234001 /test`: Send '/test' to chat ID '8671234001'.
*   `tg-signer monitor run`: Configure and start message monitoring and automated responses.
*   `tg-signer multi-run -a account_a -a account_b same_task`: Run 'same_task' configuration with 'account_a' and 'account_b'.

### Configuring Proxies

`tg-signer` does not use system proxies by default. Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line argument:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Logging In

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. This logs in to your Telegram account and retrieves your chat list. Ensure the chats you want to interact with are in the list.

### Sending a Single Message

```bash
tg-signer send-text 8671234001 hello
```

Sends 'hello' to chat ID '8671234001'.

### Running a Check-in Task

```bash
tg-signer run
```

Or run a specific task:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

**Check-in Task Configuration Example:**

```
... [Example configuration prompts for a check-in task, as provided in the original README] ...
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure a monitoring task.

**Monitoring Configuration Example:**

```
... [Example configuration prompts for a monitoring task, as provided in the original README] ...
```

## Version History

*   **0.7.6:** Fix for message forwarding in multiple chats.
*   **0.7.5:** Captures and logs RPC errors during task execution, Bump kurigram version to 2.2.7.
*   **0.7.4:** Support fixed time intervals when executing multiple actions. No longer limit daily execution when using crontab for scheduling.
*   **0.7.2:**  Added support for forwarding messages to external endpoints (UDP, HTTP).
*   **0.7.0:**  Each chat can execute multiple actions in sequence.
    *   Send Text, Send Dice, Click keyboard button, Image recognition, and Answer calculation question
*   **0.6.6:** Added support for sending DICE messages.
*   **0.6.5:** Fixed issue where check-in records were shared when running multiple accounts.
*   **0.6.4:** Added support for simple calculation questions and improved check-in configuration and message handling.
*   **0.6.3:** Compatible with the breaking changes of kurigram 2.1.38.
*   **0.6.2:** Ignore chats where sending messages fails during check-in.
*   **0.6.1:** Support for image recognition after clicking a button.
*   **0.6.0:**  Signer now supports crontab scheduling; Monitor adds 'all' rule; Monitor supports push notifications using server酱;  Added multi-run.
*   **0.5.2:** Monitor supports AI message replies. Added bulk configuration for Telegram's scheduled messages.
*   **0.5.1:** Added `import` and `export` commands for configuration management.
*   **0.5.0:**  Click keyboard by text, AI image recognition.

## Configuration and Data Storage

Configuration and data are stored in the `.signer` directory. You'll find:

```
.signer
├── latest_chats.json
├── me.json
├── monitors
│   ├── my_monitor
│   │   └── config.json
└── signs
    └── linuxdo
        ├── config.json
        └── sign_record.json

3 directories, 4 files