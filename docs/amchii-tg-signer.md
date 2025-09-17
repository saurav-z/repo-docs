# TG-Signer: Automate Telegram Tasks with Python

**Automate your Telegram experience with TG-Signer, a powerful Python tool for signing in, monitoring, and responding to messages.** Check out the original repository [here](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Sign-In:** Perform daily sign-in tasks with customizable timing and random delays.
*   **Keyboard Interaction:** Automatically click keyboard buttons based on configured text.
*   **AI-Powered Image Recognition:** Utilize AI to analyze and interact with images within Telegram.
*   **Message Monitoring and Auto-Reply:** Monitor personal chats, groups, and channels, forwarding and auto-replying to messages based on your rules.
*   **Action Flows:** Execute complex action sequences based on custom configurations.
*   **Flexible Scheduling:** Schedule tasks using either specific times or cron expressions.
*   **Multi-Account Support:** Run the same configurations across multiple Telegram accounts.
*   **Message Forwarding:** Forward matched messages to external endpoints via UDP or HTTP.

## Installation

Requires Python 3.9 or later.

```bash
pip install -U tg-signer
```

For improved performance:

```bash
pip install "tg-signer[speedup]"
```

### Docker

You can build a Docker image using the provided `Dockerfile` in the `./docker` directory and the corresponding [README](./docker/README.md).

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <subcommand> --help to view usage instructions

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
                                  overwrites the environment variable
                                  `TG_PROXY`  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path  [default: .]
  -a, --account TEXT              Custom account name, the session file name
                                  corresponds to <account>.session  [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and sign-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String,
                                  overwrites the environment variable
                                  `TG_SESSION_STRING`  [env var:
                                  TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export the configuration, defaults to output to the
                          terminal.
  import                  Import the configuration, defaults to read from the
                          terminal.
  list                    List existing configurations
  list-members            Query the members of a chat (group or channel), the
                          channel requires administrator privileges
  list-schedule-messages  Display configured timed messages
  login                   Log in to the account (used to obtain a session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts simultaneously using a set of
                          configurations
  reconfig                Reconfigure
  run                     Run sign-in based on task configuration
  run-once                Run a sign-in task once, even if the sign-in task has
                          been executed today
  schedule-messages       Batch configuration of Telegram's built-in timed
                          message sending function
  send-text               Send a message once, make sure the current session
                          has "seen" the `chat_id`
  version                 Show version
```

### Examples:

```bash
tg-signer run
tg-signer run my_sign  # Run the 'my_sign' task directly without prompting.
tg-signer run-once my_sign  # Run the 'my_sign' task once.
tg-signer send-text 8671234001 /test  # Send '/test' text to chat_id '8671234001'.
tg-signer send-text -- -10006758812 浇水  # Use POSIX style for negative numbers: '--' before '-'.
tg-signer send-text --delete-after 1 8671234001 /test  # Send '/test' and delete after 1 second.
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins.
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages for 10 days at 0:00.
tg-signer monitor run  # Configure message monitoring and auto-reply.
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' with accounts 'account_a' and 'account_b'.
```

### Configuration

#### Proxy

Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line argument.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

#### Login

Use `tg-signer login` to authenticate and retrieve a session.

#### Send Text

Send a message using `tg-signer send-text`.  Example: `tg-signer send-text 8671234001 hello`

#### Run Sign-In Tasks

Execute sign-in tasks with `tg-signer run`.  You can specify a task name directly: `tg-signer run linuxdo`.

**Example Configuration:**

```
... (Sign-in configuration example as provided in the original README, formatted for clarity) ...
```

#### Configure and Run Monitoring

Configure and run monitoring using `tg-signer monitor run`.

**Example Configuration:**

```
... (Monitoring configuration example as provided in the original README, formatted for clarity) ...
```

### Data Storage

Configurations and data are stored in the `.signer` directory.  The directory structure includes:

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
```

### Version Changelog

```
... (Changelog as provided in the original README) ...
```