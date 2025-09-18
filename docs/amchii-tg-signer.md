# tg-signer: Automate Telegram Tasks with Python 🤖

**Effortlessly automate your Telegram activities, from daily check-ins to message monitoring and AI-powered responses, with tg-signer!** [See the original repository](https://github.com/amchii/tg-signer)

## Key Features

*   ✅ **Automated Check-ins:** Schedule daily check-ins with customizable time offsets and random delays.
*   ⌨️ **Keyboard Interaction:**  Automated interaction with Telegram's in-app keyboards based on text.
*   🖼️ **AI-Powered Image Recognition:** Leverage AI to analyze images and select keyboard options.
*   💬 **Message Monitoring & Auto-Reply:**  Monitor personal chats, groups, and channels, with forwarding and automated responses.
*   🔄 **Configurable Action Flows:**  Define and execute custom action sequences for advanced automation.
*   ⏰ **Scheduled Messages:** Utilize Telegram's built-in scheduling functionality.

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

Build your own Docker image using the `Dockerfile` and `README.md` in the [docker](./docker) directory.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <SUBCOMMAND> --help to see usage for a specific command.

Subcommand Aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level: `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, relative paths are supported
                                  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, e.g.: socks5://127.0.0.1:1080,
                                  overrides the `TG_PROXY` environment variable
                                  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG sessions, relative paths
                                  are supported  [default: .]
  -a, --account TEXT              Custom account name, corresponding session file
                                  name is <account>.session  [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configurations, check-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String, overrides the
                                  `TG_SESSION_STRING` environment variable
                                  [env var: TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export configurations, defaults to output to the
                          terminal.
  import                  Import configurations, defaults to read from the
                          terminal.
  list                    List existing configurations
  list-members            Query the members of a chat (group or channel), the
                          channel requires administrator privileges
  list-schedule-messages  Display configured scheduled messages
  login                   Log in to the account (used to obtain the session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts simultaneously with one set of
                          configurations
  reconfig                Reconfigure
  run                     Run check-in based on task configuration
  run-once                Run a check-in task once, even if the check-in task
                          has already been executed today
  schedule-messages       Batch configure Telegram's built-in timed message
                          sending function
  send-text               Send a message once, please make sure the current
                          session has "seen" the `chat_id`
  version                 Show version
```

**Examples:**

```bash
# Run a check-in task
tg-signer run

# Run a specific task directly
tg-signer run my_sign

# Run a check-in task once
tg-signer run-once my_sign

# Send a text message
tg-signer send-text 8671234001 /test

# List group members
tg-signer list-members --chat_id -1001680975844 --admin

# Schedule messages with crontab
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好

# Configure and run message monitoring
tg-signer monitor run

# Run the same task across multiple accounts
tg-signer multi-run -a account_a -a account_b same_task
```

## Configuration

### Configure Proxy (If Needed)

Use the environment variable `TG_PROXY` or the command-line argument `--proxy`:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in and retrieve your chat list. Ensure the chats you want to interact with are in this list.

### Send a Message

```bash
tg-signer send-text 8671234001 hello
```

### Run Check-in Tasks

```bash
tg-signer run
```

or specify a task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the check-in actions.

### Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your monitoring rules.

## Data and Configuration Storage

Data and configurations are stored in the `.signer` directory. You can view the structure with `tree .signer`:

```
.signer
├── latest_chats.json  # Recent chat list
├── me.json  # User information
├── monitors  # Monitoring configurations
│   ├── my_monitor  # Monitor task name
│   │   └── config.json  # Monitor configuration
└── signs  # Check-in task configurations
    └── linuxdo  # Check-in task name
        ├── config.json  # Check-in configuration
        └── sign_record.json  # Check-in records

3 directories, 4 files