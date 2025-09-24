# tg-signer: Automate Your Telegram Tasks

**Automate your Telegram interactions with `tg-signer`, a powerful tool for daily check-ins, message monitoring, and automated replies!** Check out the project on [GitHub](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Sign-ins:** Schedule daily check-ins with random delays.
*   **Keyboard Interaction:** Click buttons based on configured text.
*   **AI-Powered Image Recognition:** Automate actions by recognizing and interacting with images.
*   **Message Monitoring & Auto-Reply:** Monitor, forward, and automatically reply to messages in personal chats, groups, and channels.
*   **Customizable Action Flows:** Execute complex actions based on your configurations.
*   **Scheduled Message Sending:** Leverage Telegram's built-in scheduling for automated messages.
*   **Multi-Account Support:** Run the same task on multiple accounts simultaneously.

## Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For improved performance:

```bash
pip install "tg-signer[speedup]"
```

### Docker

Build your own image using the provided [Dockerfile](./docker) and [README](./docker/README.md) in the `docker` directory.

## Usage

```text
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <子命令> --help to view usage instructions

Subcommand aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, can be a relative path  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, for example: socks5://127.0.0.1:1080,
                                  overrides the environment variable `TG_PROXY`  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a relative path  [default: .]
  -a, --account TEXT              Custom account name, corresponding session file name is <account>.session  [env
                                  var: TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store configurations and check-in records, etc.  [default:
                                  .signer]
  --session-string TEXT           Telegram Session String,
                                  overrides the environment variable `TG_SESSION_STRING`  [env var:
                                  TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory, defaults to False, stored in file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to terminal.
  import                  Import configuration, defaults to read from terminal.
  list                    List existing configurations
  list-members            Query members of a chat (group or channel), channels require administrator privileges
  list-schedule-messages  Show scheduled messages
  login                   Login to the account (used to get session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts simultaneously using one set of configurations
  reconfig                Reconfigure
  run                     Run check-ins according to the task configuration
  run-once                Run a check-in task once, even if the check-in task has been executed today
  schedule-messages       Batch configure Telegram's built-in timed messaging function
  send-text               Send a message once, please make sure the current session has "seen" the `chat_id`
  version                 Show version
```

Examples:

```bash
tg-signer run
tg-signer run my_sign  # Run the 'my_sign' task directly without prompting
tg-signer run-once my_sign  # Run the 'my_sign' task directly once
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id '8671234001'
tg-signer send-text -- -10006758812 浇水  # For negative numbers, use POSIX style with '--' before the '-'
tg-signer send-text --delete-after 1 8671234001 /test  # Send '/test' to chat_id '8671234001' and delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Send a message to '-1001680975844' at 0:00 for the next 10 days
tg-signer monitor run  # Configure personal, group, and channel message monitoring and auto-reply
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' config with both 'account_a' and 'account_b' accounts
```

### Configuring a Proxy (if needed)

`tg-signer` does not read the system proxy. Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option.

Example:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Logging In

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in. This will retrieve your recent chat list, ensuring the chats you want to interact with are visible.

### Sending a Message

```bash
tg-signer send-text 8671234001 hello  # Send 'hello' to chat_id '8671234001'
```

### Running a Sign-in Task

```bash
tg-signer run
```

Or, specify the task name directly:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

#### Example:

```text
... (configuration prompts) ...
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your monitoring task.

#### Example:

```text
... (configuration prompts) ...
```

#### Example Explanation:

1.  Chat ID and user ID support both integer IDs and usernames (usernames start with @).
2.  Matching Rules:
    *   `exact`: Exact message match.
    *   `contains`: Message contains the specified text (case-insensitive).
    *   `regex`: Regular expression matching (see [Python regex documentation](https://docs.python.org/zh-cn/3/library/re.html)).
3.  Message Structure (example):

```json
... (example message JSON) ...
```

#### Example Run Output:

```text
... (example run output) ...
```

### Version Changelog

... (Changelog entries from original README) ...

### Configuration and Data Storage

Data and configurations are stored in the `.signer` directory. Running `tree .signer` will show:

```text
.signer
├── latest_chats.json  # Recently obtained dialogues
├── me.json  # Personal Information
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitor task name
│       └── config.json  # Monitor configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in record

3 directories, 4 files
```