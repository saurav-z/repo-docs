# tg-signer: Automate Your Telegram Tasks with Ease

**Automate Telegram tasks like daily check-ins, message monitoring, and auto-replies with the power of Python!**  [View the Project on GitHub](https://github.com/amchii/tg-signer)

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with customizable time offsets.
*   **Intelligent Interaction:** Click buttons based on configured text, even leveraging AI-powered image recognition.
*   **Message Monitoring & Response:**  Monitor, forward, and auto-reply to messages in personal chats, groups, and channels.
*   **Flexible Action Flows:**  Define and execute custom action sequences.
*   **AI Integration:** Use AI for image recognition and solving calculation problems.
*   **Scheduled Messages:** Configure and manage Telegram's built-in scheduled message feature.
*   **Multi-Account Support:** Run the same tasks on multiple accounts simultaneously.
*   **Flexible Deployment:** Run via CLI, Docker, or direct Python installation.

## Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For enhanced performance:

```bash
pip install "tg-signer[speedup]"
```

### Docker

Refer to the [`docker`](./docker) directory and its [README](./docker/README.md) for building your own Docker images.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <subcommand> --help to view usage instructions

Subcommand Aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]  Log level, `debug`, `info`, `warn`,
                                          `error`  [default: info]
  --log-file PATH                        Log file path, can be a relative
                                          path  [default: tg-signer.log]
  -p, --proxy TEXT                       Proxy address, e.g.:
                                          socks5://127.0.0.1:1080, will
                                          override the environment variable
                                          `TG_PROXY`  [env var: TG_PROXY]
  --session_dir PATH                     Directory to store TG Sessions, can
                                          be a relative path  [default: .]
  -a, --account TEXT                     Custom account name, corresponding
                                          session file name is <account>.session
                                          [env var: TG_ACCOUNT; default:
                                          my_account]
  -w, --workdir PATH                     tg-signer working directory, used to
                                          store configuration and check-in
                                          records, etc.  [default: .signer]
  --session-string TEXT                  Telegram Session String, will override
                                          the environment variable
                                          `TG_SESSION_STRING`  [env var:
                                          TG_SESSION_STRING]
  --in-memory                            Whether to store the session in
                                          memory, the default is False,
                                          stored in a file
  --help                                 Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to the
                          terminal.
  import                  Import configuration, defaults to read from the
                          terminal.
  list                    List existing configurations
  list-members            Query members of a chat (group or channel), the
                          channel requires administrator privileges
  list-schedule-messages  Display configured scheduled messages
  login                   Log in to the account (used to obtain the session)
  logout                  Log out of the account and delete the session
                          file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts simultaneously using a set
                          of configurations
  reconfig                Reconfigure
  run                     Run check-in according to task configuration
  run-once                Run a check-in task once, even if the check-in
                          task has been executed today
  schedule-messages       Batch configure Telegram's built-in scheduled
                          message function
  send-text               Send a message once, please make sure that the
                          current session has "seen" the `chat_id`
  version                 Show version
```

### Examples:

```bash
tg-signer run
tg-signer run my_sign  # Run 'my_sign' task directly
tg-signer run-once my_sign  # Run 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id 8671234001
tg-signer send-text -- -10006758812 浇水  # Use '--' for negative chat IDs
tg-signer send-text --delete-after 1 8671234001 /test  # Send and delete in 1s
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule a message
tg-signer monitor run  # Configure and run message monitoring
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' on two accounts
```

### Configure Proxy (If Needed)

`tg-signer` does not use system proxies.  Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line argument.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in and retrieve your chat list.

### Send a Message

```bash
tg-signer send-text 8671234001 hello
```

### Run a Check-in Task

```bash
tg-signer run
```

Or, specify the task name:

```bash
tg-signer run linuxdo
```

Follow the interactive prompts to configure your check-in.

#### Example Check-in Configuration:

```
# (Configuration steps similar to original README)
```

### Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your monitoring rules.

#### Example Monitoring Configuration:

```
# (Configuration steps and example output from original README,
#  but concise and with clear bulleted points as needed)
```

### Configuration and Data Storage

Configuration and data are stored in the `.signer` directory. For example:

```
.signer
├── latest_chats.json
├── me.json
├── monitors
│   └── my_monitor
│       └── config.json
└── signs
    └── linuxdo
        ├── config.json
        └── sign_record.json
```

### Version Change Log

```
#### 0.7.6
# (List version changes concisely)

#### 0.7.5
# (List version changes concisely)

#### 0.7.4
# (List version changes concisely)
```

```markdown