# Automate Your Telegram Tasks with tg-signer

**Effortlessly automate your Telegram interactions with tg-signer, a powerful Python tool for daily check-ins, message monitoring, and automated responses. [View the original repository on GitHub](https://github.com/amchii/tg-signer)**

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with customizable timing and random error offsets.
*   **Keyboard Interaction:** Automate actions by clicking keyboard buttons based on configured text.
*   **AI-Powered Image Recognition:** Leverage AI for image recognition and automated keyboard interactions.
*   **Message Monitoring & Response:** Monitor personal chats, groups, and channels; forward messages; and set up automated replies.
*   **Action Flow Execution:** Configure action sequences to execute complex tasks.
*   **Flexible Configuration:** Easily configure proxy settings, account names, working directories, and session management.
*   **Docker Support:** Build and run the application using Docker.
*   **Multi-Account Management:** Run multiple accounts concurrently using a single configuration.
*   **Scheduled Messages:** Configure Telegram's built-in scheduling for automated message sending.
*   **Message Forwarding:** Forward messages to external destinations via UDP and HTTP.

## Getting Started

### Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For improved performance:

```bash
pip install "tg-signer[speedup]"
```

### Docker

Build your own image using the provided [Dockerfile](./docker) and read the [README](./docker/README.md) in the `docker` directory.

### Usage

Use the command-line interface to manage your Telegram automation.

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <subcommand> --help for usage instructions.

Subcommand aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, can be a relative path
                                  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, e.g.: socks5://127.0.0.1:1080,
                                  overrides the environment variable
                                  `TG_PROXY`  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path  [default: .]
  -a, --account TEXT              Custom account name, corresponding session
                                  file name is <account>.session  [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and check-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String, overrides the
                                  environment variable `TG_SESSION_STRING`
                                  [env var: TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  default is False, store in file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to the
                          terminal.
  import                  Import configuration, defaults to reading from the
                          terminal.
  list                    List existing configurations
  list-members            Query members of a chat (group or channel), channel
                          requires administrator privileges
  list-schedule-messages  Display scheduled messages
  login                   Login to account (used to get session)
  logout                  Logout account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts with one set of configurations
  reconfig                Reconfigure
  run                     Run check-in based on task configuration
  run-once                Run a check-in task once, even if the check-in task
                          has been executed today
  schedule-messages       Configure Telegram's built-in scheduled message
                          feature in batches
  send-text               Send a message once, make sure the current session
                          has "seen" the `chat_id`
  version                 Show version
```

#### Examples

```bash
tg-signer run
tg-signer run my_sign  # Run the 'my_sign' task directly without asking
tg-signer run-once my_sign  # Run the 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id 8671234001
tg-signer send-text -- -10006758812 浇水  # Use '--' for negative chat IDs
tg-signer send-text --delete-after 1 8671234001 /test  # Send '/test' and delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages for 10 days at 0:00
tg-signer monitor run  # Configure message monitoring and auto-reply
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' with 'account_a' and 'account_b'
```

### Configure Proxy (if needed)

`tg-signer` does not read the system proxy. Configure the proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. This will log you in and fetch your recent chats. Ensure the chat you want to interact with is in the list.

### Send a Message

```bash
tg-signer send-text 8671234001 hello  # Sends 'hello' to chat_id 8671234001
```

### Run a Check-in Task

```bash
tg-signer run
```

Or run a predefined task:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

#### Example:

(The example is truncated. Please refer to the original README for the full configuration process)

### Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure the monitoring task.

#### Example:

(The example is truncated. Please refer to the original README for the full configuration process.)

### Version Change Log

(The version change log is truncated. Please refer to the original README for the full version change log.)

### Configuration and Data Storage Location

Data and configuration are stored in the `.signer` directory. Here's the typical file structure:

```
.signer
├── latest_chats.json  # Recent chats
├── me.json  # Personal information
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitor task name
│   │   └── config.json  # Monitor configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in record

3 directories, 4 files