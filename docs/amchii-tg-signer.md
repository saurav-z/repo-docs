# tg-signer: Automate Telegram Tasks with Python

**Automate your Telegram interactions with `tg-signer`, offering features like scheduled check-ins, message monitoring, and AI-powered responses.** ([See the original repo](https://github.com/amchii/tg-signer))

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with customizable time offsets.
*   **Keyboard Interaction:** Configure the bot to automatically click buttons based on text or image recognition.
*   **Message Monitoring & Auto-Reply:** Monitor personal chats, groups, and channels, and set up automated responses and forwarding.
*   **AI Integration:** Leverage AI for image recognition, answering calculation questions, and responding to messages.
*   **Flexible Configuration:** Supports a variety of actions, including sending text, dice emojis, and deleting messages after a set time.
*   **Multi-Account Support:** Easily manage multiple Telegram accounts using a single configuration.
*   **Extensive Commands:** Execute a wide array of tasks, from running check-ins to scheduling messages, with simple command-line arguments.

## Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For performance improvements:

```bash
pip install "tg-signer[speedup]"
```

### Docker

While a pre-built image isn't available, you can build your own using the provided `Dockerfile` and [README](./docker/README.md) in the `docker` directory.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <subcommand> --help for usage instructions

Subcommand Aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, can be a relative path
                                  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, for example:
                                  socks5://127.0.0.1:1080, overrides the
                                  environment variable `TG_PROXY`
                                  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path  [default: .]
  -a, --account TEXT              Custom account name, the session file name
                                  corresponds to <account>.session  [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and check-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String, overrides the
                                  environment variable `TG_SESSION_STRING`
                                  [env var: TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory, the
                                  default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, default output to terminal.
  import                  Import configuration, default read from terminal.
  list                    List existing configurations
  list-members            Query chat (group or channel) members, channels
                          require administrator privileges
  list-schedule-messages  Show configured scheduled messages
  login                   Log in to the account (used to get the session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts with one set of configurations
  reconfig                Reconfigure
  run                     Run check-in according to task configuration
  run-once                Run a check-in task once, even if the check-in task
                          has been executed today
  schedule-messages       Batch configure Telegram's built-in timed message
                          sending function
  send-text               Send a message once, make sure the current session
                          has "seen" the `chat_id`
  version                 Show version
```

### Examples

```bash
tg-signer run                       # Run a configured check-in task.
tg-signer run my_sign               # Run a specific check-in task directly.
tg-signer run-once my_sign          # Run a check-in task once, regardless of previous runs.
tg-signer send-text 8671234001 /test  # Send text to a chat ID.
tg-signer send-text -- -10006758812 浇水 # Send text to a channel, using -- to handle negative IDs.
tg-signer send-text --delete-after 1 8671234001 /test # Send text to a chat ID and delete it after 1 second.
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins.
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好 # Schedule messages using cron.
tg-signer monitor run               # Configure message monitoring.
tg-signer multi-run -a account_a -a account_b same_task # Run a task with multiple accounts.
```

### Configuring a Proxy (If Needed)

`tg-signer` does not use the system proxy. Configure a proxy using the environment variable `TG_PROXY` or the `--proxy` command-line argument:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Logging In

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in and retrieve your recent chat list. Ensure the chat you want to use for check-ins is in the list.

### Sending a Message

```bash
tg-signer send-text 8671234001 hello
```

### Running a Check-in Task

```bash
tg-signer run
```

Or specify a task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

#### Example Check-in Configuration:

```
# Example configuration showing the various action types.
# 1: Send Text
# 3: Click on a Keyboard Button
# 4: Respond to Image Recognition
# 5: Solve a Calculation Question
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your monitoring rules.

#### Example Monitoring Configuration:

```
# Example configuration
```

### Version Changelog

```
# Version Change Logs
```

### Configuration and Data Storage

Configuration and data are stored in the `.signer` directory. Here's how it's organized:

```
.signer
├── latest_chats.json  # Recent conversations
├── me.json  # User information
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitoring task name
│       └── config.json  # Monitoring configuration
└── signs  # Check-in tasks
    └── linuxdo  # Check-in task name
        ├── config.json  # Check-in configuration
        └── sign_record.json  # Check-in record

3 directories, 4 files