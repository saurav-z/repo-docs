# tg-signer: Automate Telegram Tasks with Ease

**Automate your Telegram activities with tg-signer, a powerful Python tool for daily check-ins, message monitoring, and automated replies.  [View the original repository](https://github.com/amchii/tg-signer).**

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with customizable time variations.
*   **Keyboard Interaction:** Automate button clicks based on configured text.
*   **AI-Powered Image Recognition:**  Utilize AI to identify and click buttons based on image content.
*   **Message Monitoring & Response:** Monitor, forward, and automatically reply to messages in personal chats, groups, and channels.
*   **Action Flows:** Execute complex workflows based on configurable actions.
*   **Flexible Configuration:** Configure via command-line arguments, environment variables, or configuration files.
*   **Multi-Account Support:**  Manage multiple Telegram accounts simultaneously.
*   **Message Scheduling:** Leverage Telegram's built-in message scheduling.
*   **Advanced Monitoring:**  Includes regex and AI-based response options.
*   **External Integrations:** Forward messages to UDP and HTTP endpoints.

## Installation

Requires Python 3.9 or higher.

Install using pip:

```bash
pip install -U tg-signer
```

For faster performance, install with speedup:

```bash
pip install "tg-signer[speedup]"
```

### Docker

Build your own Docker image using the provided [Dockerfile](./docker/Dockerfile) and refer to the [README](./docker/README.md) in the `docker` directory.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <subcommand> --help to see usage instructions

Subcommand Aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, which can be a relative
                                  path  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, e.g.: socks5://127.0.0.1:1080,
                                  overrides the environment variable
                                  `TG_PROXY`  [env var: TG_PROXY]
  --session_dir PATH              Directory for storing TG Sessions, which
                                  can be a relative path  [default: .]
  -a, --account TEXT              Custom account name, the session file name
                                  corresponds to <account>.session  [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and check-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String,
                                  overrides the environment variable
                                  `TG_SESSION_STRING`  [env var:
                                  TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  the default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export the configuration, defaulting to output to
                          the terminal.
  import                  Import the configuration, defaulting to read from
                          the terminal.
  list                    List existing configurations
  list-members            Query members of a chat (group or channel), the
                          channel requires administrator privileges
  list-schedule-messages  Display configured scheduled messages
  login                   Login to the account (used to get session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts simultaneously using one
                          configuration
  reconfig                Reconfigure
  run                     Run check-in based on task configuration
  run-once                Run a check-in task once, even if the check-in task
                          has already been performed today
  schedule-messages       Configure Telegram's built-in scheduled message
                          functionality in batches
  send-text               Send a message once, make sure the current session
                          has "seen" the `chat_id`
  version                 Show version
```

### Examples

```bash
tg-signer run  # Runs a configured check-in task.
tg-signer run my_sign  # Runs the 'my_sign' task directly without prompts.
tg-signer run-once my_sign  # Runs 'my_sign' once, even if already run today.
tg-signer send-text 8671234001 /test  # Sends '/test' to chat ID 8671234001.
tg-signer send-text -- -10006758812 water  # Uses -- for negative chat IDs.
tg-signer send-text --delete-after 1 8671234001 /test # Sends '/test' then deletes in 1 sec.
tg-signer list-members --chat_id -1001680975844 --admin # Lists channel admins.
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 Hello  # Schedules "Hello"
tg-signer monitor run  # Configures and runs message monitoring.
tg-signer multi-run -a account_a -a account_b same_task # Runs same task on multiple accounts.
```

### Configuring a Proxy (If Needed)

`tg-signer` does not read system proxies.  Use the `TG_PROXY` environment variable or the `--proxy` command-line option:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Logging In

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. This will log you in and retrieve your recent chat list. Ensure the chat you want to interact with is in the list.

### Sending a Single Message

```bash
tg-signer send-text 8671234001 hello  # Sends 'hello' to chat ID 8671234001.
```

### Running a Check-in Task

```bash
tg-signer run
```

Or, specify the task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

### Configuration and Data Storage

Data and configurations are stored in the `.signer` directory by default.

```
.signer
├── latest_chats.json  # Recent chats
├── me.json  # Personal information
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitor task name
│       └── config.json  # Monitor configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in records

3 directories, 4 files