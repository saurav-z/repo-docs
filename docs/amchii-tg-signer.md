# tg-signer: Automate Your Telegram Tasks with Python

**Tired of manual Telegram tasks?** tg-signer is a versatile Python-based tool that automates daily check-ins, monitors messages, and responds to interactions, simplifying your Telegram experience. 🔗 [View the Project on GitHub](https://github.com/amchii/tg-signer)

## Key Features

*   **Automated Check-ins:** Schedule and execute Telegram check-ins with customizable timing and random delays.
*   **Intelligent Interaction:** Click keyboard buttons based on configured text or leverage AI-powered image recognition for automated actions.
*   **Message Monitoring & Auto-Response:**  Monitor personal chats, groups, and channels, with options for forwarding and automatic replies based on custom rules.
*   **Flexible Action Flows:**  Define action sequences for complex tasks, including sending text, clicking buttons, and more.
*   **Multi-Account Support:**  Easily manage multiple Telegram accounts with simultaneous execution.
*   **Advanced Configuration:** Customize tasks using crontab expressions for precise scheduling, configure AI integration, and define message forwarding.
*   **Powerful Monitoring Capabilities:** Set up rules to react to specific keywords, user IDs, or message patterns.  Get notified via Server酱 or forward messages to external endpoints (UDP/HTTP).

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

Build your own Docker image using the provided [Dockerfile](./docker/Dockerfile) and  [README](./docker/README.md) in the `docker` directory.

## Usage

### Command-Line Interface

Use the following commands to manage and automate your Telegram activities:

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
                                  overrides the environment variable
                                  `TG_PROXY` [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path [default: .]
  -a, --account TEXT              Custom account name, corresponding session
                                  file name is <account>.session [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and check-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String, overrides the
                                  environment variable `TG_SESSION_STRING`
                                  [env var: TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, default output to terminal.
  import                  Import configuration, default read from terminal.
  list                    List existing configurations
  list-members            Query the members of a chat (group or channel),
                          channel requires administrator privileges
  list-schedule-messages  Show configured scheduled messages
  login                   Login to the account (used to get the session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts simultaneously using one set
                          of configurations
  reconfig                Reconfigure
  run                     Run check-in according to task configuration
  run-once                Run a check-in task once, even if the check-in task
                          has been executed today
  schedule-messages       Configure Telegram's built-in scheduled message
                          function in batches
  send-text               Send a message once, make sure the current
                          session has "seen" this `chat_id`
  version                 Show version
```

### Examples

```bash
tg-signer run  # Run a configured check-in task
tg-signer run my_sign  # Run the 'my_sign' task directly
tg-signer run-once my_sign  # Run 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id 8671234001
tg-signer send-text -- -10006758812 浇水 # Use POSIX style for negative numbers
tg-signer send-text --delete-after 1 8671234001 /test  # Send and delete in 1s
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好 # Schedule messages
tg-signer monitor run  # Configure message monitoring and auto-reply
tg-signer multi-run -a account_a -a account_b same_task  # Run same_task config with multiple accounts
```

### Configuration

#### Proxy (if needed)

`tg-signer` does not read system proxies. Configure it using the `TG_PROXY` environment variable or the `--proxy` command-line option:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

#### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in.  This will also retrieve your chat list, ensure the target chats are present.

#### Send a Single Message

```bash
tg-signer send-text 8671234001 hello  # Send 'hello' to chat_id 8671234001
```

#### Run a Check-in Task

```bash
tg-signer run
```

Or, specify the task name directly:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

##### Example Check-in Configuration

```
... (configuration example as provided in original README) ...
```

#### Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to set up your monitoring rules.

##### Example Monitoring Configuration

```
... (configuration example as provided in original README) ...
```

##### Example Explanation of Monitoring

1.  **Chat ID and User ID:** Both integer IDs and usernames (starting with `@`) are supported.  Remember that usernames may not always exist.
2.  **Matching Rules:**

    *   `exact`:  Exact match of the message content.
    *   `contains`: Message *contains* the specified text (case-insensitive).
    *   `regex`:  Matches based on a regular expression.  See [Python regex documentation](https://docs.python.org/zh-cn/3/library/re.html).
    *   `all`: Matches any message.
3.  **Message Structure:**

    ```json
    {
    ... (Message Structure example as provided in original README) ...
    }
    ```

##### Example Monitoring Output

```
... (Example monitoring output as provided in original README) ...
```

### Version Changelog

```
... (Version Changelog as provided in original README) ...
```

### Configuration and Data Storage

Configurations and data are stored in the `.signer` directory.  The directory structure after running `tree .signer` looks like this:

```
... (Directory Structure example as provided in original README) ...
```