# tg-signer: Automate Telegram Tasks with Python

**Tired of manual Telegram tasks?** Automate your Telegram activities with `tg-signer`, a powerful Python tool for scheduled sign-ins, message monitoring, and automated responses.  [View the original repository here](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Sign-Ins:** Schedule daily sign-ins with randomized time offsets.
*   **Interactive Automation:**  Click buttons based on configured text and leverage AI for image recognition and button clicks.
*   **Message Monitoring & Response:** Monitor, forward, and automatically reply to messages in personal chats, groups, and channels.
*   **Action Flow Execution:** Execute custom action flows based on your configurations.
*   **Flexible Deployment:** Supports direct installation, optional speedup, and Docker.
*   **Configurable:**  Easily configure proxies, accounts, working directories, and sessions.

## Installation

Requires Python 3.9 or higher.

**Install with pip:**

```bash
pip install -U tg-signer
```

**For optimized performance:**

```bash
pip install "tg-signer[speedup]"
```

**Docker (Build your own image):**

See the [docker](./docker) directory for the Dockerfile and [README](./docker/README.md).

## Usage

```bash
tg-signer [OPTIONS] COMMAND [ARGS]...
```

Use `<subcommand> --help` to view specific command usage.

**Subcommand Aliases:**

*   `run_once` -> `run-once`
*   `send_text` -> `send-text`

**Available Commands:**

*   `export`: Exports configurations to the terminal.
*   `import`: Imports configurations from the terminal.
*   `list`: Lists existing configurations.
*   `list-members`: Lists members of a chat (group or channel). Requires admin permissions for channels.
*   `list-schedule-messages`: Displays configured scheduled messages.
*   `login`: Logs into your Telegram account (required for session acquisition).
*   `logout`: Logs out and deletes the session file.
*   `monitor`: Configure and run message monitoring.
*   `multi-run`: Run the same configuration across multiple accounts simultaneously.
*   `reconfig`: Reconfigure existing settings.
*   `run`: Runs a sign-in task based on configuration.
*   `run-once`: Runs a sign-in task once, regardless of whether it has already run today.
*   `schedule-messages`: Configures Telegram's built-in scheduled message feature.
*   `send-text`: Sends a text message to a specified chat.
*   `version`: Displays the tool's version.

**Example Usage:**

```bash
tg-signer run  # Run configured sign-in tasks
tg-signer run my_sign  # Run 'my_sign' task immediately, without prompts
tg-signer run-once my_sign  # Run 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat ID '8671234001'
tg-signer send-text -- -10006758812 浇水 # Send "浇水" to a chat with negative ID
tg-signer list-members --chat_id -1001680975844 --admin # List admins of a channel
tg-signer monitor run  # Configure and run message monitoring with auto-replies
tg-signer multi-run -a account_a -a account_b same_task  # Run same_task with multiple accounts
```

## Configuration

### Proxy Configuration

`tg-signer` does not read system proxies directly.  Configure proxies using the `TG_PROXY` environment variable or the `--proxy` command-line option.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code.  This will log you in and retrieve recent chats.

### Sending a Message

```bash
tg-signer send-text 8671234001 hello  # Send 'hello' to chat ID 8671234001
```

### Running Sign-in Tasks

```bash
tg-signer run
```

Or, specify the task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure your sign-in.

## Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your monitoring rules.

## Version Change Log

*   **(0.7.6)** Fixed issue with message forwarding in multiple chats.
*   **(0.7.5)**  Improved error handling and dependency updates.
*   **(0.7.4)** Added fixed-interval support for multiple actions and removed daily execution limit for cron-based scheduling.
*   **(0.7.2)** Added forwarding to external endpoints (UDP, HTTP).
*   **(0.7.0)** Implemented ordered execution of multiple actions for each chat, including text, dice, button clicks, image selection, and math problems.
*   **(0.6.6)**  Added support for sending DICE messages.
*   **(0.6.5)**  Fixed a bug that caused sign-in records to be shared when running the same configuration across multiple accounts.
*   **(0.6.4)**  Added support for simple math problem solving and improved sign-in configuration and message handling.
*   **(0.6.3)**  Compatibility fix for breaking changes in kurigram 2.1.38.
*   **(0.6.2)**  Added handling for sign-in failures in chats.
*   **(0.6.1)**  Added support for AI-based image selection after button presses.
*   **(0.6.0)**  Enabled cron-based scheduling, "all" rule for monitoring, and server push for matching messages. Added "multi-run" for running same config for multiple accounts.
*   **(0.5.2)**  Enabled AI-powered message replies in monitoring and the feature to batch configure Telegram's scheduled messages.
*   **(0.5.1)**  Added import and export configuration commands.
*   **(0.5.0)**  Added button click automation, AI-based image recognition.

## Configuration and Data Storage

Configurations and data are stored in the `.signer` directory. The structure looks like this:

```
.signer
├── latest_chats.json  # Recent chat history
├── me.json  # User information
├── monitors  # Monitoring configurations
│   ├── my_monitor  # Monitor task name
│       └── config.json  # Monitor configuration
└── signs  # Sign-in task configurations
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in record

3 directories, 4 files