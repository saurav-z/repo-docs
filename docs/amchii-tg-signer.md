# tg-signer: Automate Telegram Tasks with Python

**Automate your Telegram activities with ease!** tg-signer is a powerful Python tool for automating tasks such as daily check-ins, message monitoring, and auto-replying. Check out the original repo [here](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Check-ins:** Schedule daily or randomized check-ins with customizable actions.
*   **Keyboard Interaction:** Automatically click buttons based on configured text or AI-powered image recognition.
*   **Message Monitoring & Auto-Reply:** Monitor personal chats, groups, and channels, with the ability to forward and automatically respond to messages.
*   **Customizable Action Flows:** Configure complex workflows to execute various actions in a sequence.
*   **Flexible Configuration:** Use command-line options, environment variables, and configuration files for easy setup.
*   **Docker Support:** Easily deploy using Docker for consistent and reproducible environments.
*   **AI Integration:** Leverage AI for image recognition and answering calculation questions.
*   **Message Forwarding:** Forward messages to external endpoints via UDP or HTTP.
*   **Scheduled Messages:** Configure and manage Telegram's built-in scheduled message feature.

## Installation

Requires Python 3.9 or higher.

**Install using pip:**

```bash
pip install -U tg-signer
```

**For improved speed:**

```bash
pip install "tg-signer[speedup]"
```

**Docker (Optional):**

Build your own Docker image using the `Dockerfile` and README in the [docker](./docker) directory.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <subcommand> --help to view usage instructions.

Subcommand aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, can be a relative path
                                  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, for example:
                                  socks5://127.0.0.1:1080, will overwrite
                                  the environment variable `TG_PROXY`  [env
                                  var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path  [default: .]
  -a, --account TEXT              Custom account name, the corresponding
                                  session file name is <account>.session  [env
                                  var: TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and check-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String, will overwrite the
                                  environment variable `TG_SESSION_STRING`
                                  [env var: TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to terminal.
  import                  Import configuration, defaults to read from terminal.
  list                    List existing configurations
  list-members            Query chat (group or channel) members, channel
                          requires administrator permissions
  list-schedule-messages  Display configured scheduled messages
  login                   Log in to the account (used to get the session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts at the same time using one set
                          of configurations
  reconfig                Reconfigure
  run                     Run check-in based on task configuration
  run-once                Run a check-in task once, even if the check-in task
                          has been executed today
  schedule-messages       Configure Telegram's built-in scheduled message
                          function in batches
  send-text               Send a message once, please make sure that the
                          current session has "seen" the `chat_id`
  version                 Show version
```

**Example Commands:**

```bash
tg-signer run
tg-signer run my_sign  # Run 'my_sign' task directly without prompting.
tg-signer run-once my_sign  # Run 'my_sign' task once.
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id '8671234001'.
tg-signer send-text -- -10006758812 浇水  # Use '--' for negative chat IDs.
tg-signer send-text --delete-after 1 8671234001 /test  # Send and delete in 1s.
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins.
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages.
tg-signer monitor run  # Configure message monitoring and auto-reply.
tg-signer multi-run -a account_a -a account_b same_task  # Run same task on multiple accounts.
```

## Configuration

### Proxy Configuration (If Needed)

Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command option.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. This process retrieves your recent chats to ensure you can interact with the correct groups/channels.

### Sending a Text Message

```bash
tg-signer send-text 8671234001 hello  # Send 'hello' to chat_id '8671234001'.
```

### Running a Check-in Task

```bash
tg-signer run
```

Or, specify a task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

**Example Check-in Configuration:**

```
# ... (Configuration prompts)
╔════════════════════════════════════════════════╗
║ Chat ID: 7661096533                            ║
║ Name: jerry bot                                ║
║ Delete After: 10                               ║
╟────────────────────────────────────────────────╨
║ Actions Flow:                                  ║
║ 1. [发送普通文本] Text: checkin                ║
║ 2. [根据文本点击键盘] Click: 签到              ║
║ 3. [根据图片选择选项]                          ║
║ 4. [回复计算题]                                ║
║ 5. [发送Dice类型的emoji] Dice: 🎲              ║
╚════════════════════════════════════════════════╝
# ... (More prompts)
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure a monitoring task.

**Example Monitoring Configuration:**

```
# ... (Configuration prompts)
配置第1个监控项
1. Chat ID（登录时最近对话输出中的ID）: -4573702599
2. 匹配规则('exact', 'contains', 'regex', 'all'): contains
3. 规则值（不可为空）: kfc
4. 只匹配来自特定用户ID的消息（多个用逗号隔开, 匹配所有用户直接回车）: @neo
5. 默认发送文本: V Me 50
6. 从消息中提取发送文本的正则表达式:
7. 等待N秒后删除签到消息（发送消息后等待进行删除, '0'表示立即删除, 不需要删除直接回车）, N:
继续配置？(y/N)：y

配置第2个监控项
1. Chat ID（登录时最近对话输出中的ID）: -4573702599
2. 匹配规则('exact', 'contains', 'regex'): regex
3. 规则值（不可为空）: 参与关键词：「.*?」
4. 只匹配来自特定用户ID的消息（多个用逗号隔开, 匹配所有用户直接回车）: 61244351
5. 默认发送文本:
6. 从消息中提取发送文本的正则表达式: 参与关键词：「(?P<keyword>(.*?))」\n
7. 等待N秒后删除签到消息（发送消息后等待进行删除, '0'表示立即删除, 不需要删除直接回车）, N: 5
继续配置？(y/N)：y

# ... (More prompts)
```

**Monitoring Example Explanation:**

1.  **Chat ID/Username:** Chat IDs and Usernames are supported, Usernames must start with `@`.
2.  **Matching Rules:**
    *   `exact`: Exact match.
    *   `contains`: Message contains the specified value (case-insensitive).
    *   `regex`: Regular expression matching (see [Python regex documentation](https://docs.python.org/zh-cn/3/library/re.html)).
    *   `all`: Matches all messages.
3.  **User Filtering:** Filter messages by specific user IDs or usernames.
4.  **Default Text:**  Send a default text if a message matches the rule.
5.  **Regex for Text Extraction:** Use regex to extract specific text from a message for auto-replying (using parentheses to capture).

## Data Storage

Configuration and data are stored in the `.signer` directory.

```bash
.signer
├── latest_chats.json  # Recent chats
├── me.json  # User info
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitor task name
│       └── config.json  # Monitor config
└── signs  # Check-in tasks
    └── linuxdo  # Check-in task name
        ├── config.json  # Check-in config
        └── sign_record.json  # Check-in records
```

## Version Changelog

**(Condensed for brevity - Refer to original for full changelog)**

*   **0.7.6:** Fix for forwarding to multiple chats,
*   **0.7.5:** Catch all RPC errors.
*   **0.7.4:** Interval for actions, cronjob fixes.
*   **0.7.2:**  Message forwarding (UDP, HTTP).
*   **0.7.0:** Multi-action flows (text, dice, keyboard, image, calc).
*   **0.6.6:** Added support for DICE messages.
*   **0.6.5:** Fixed multi-account configuration issues.
*   **0.6.4:** Added calculation support.
*   **0.6.3:**  Compatibility with Kurigram.
*   **0.6.2:** Ignore failed check-in chats.
*   **0.6.1:** Image recognition after button click.
*   **0.6.0:** Crontab scheduling, AI-powered replies, server push, `multi-run`.
*   **0.5.2:** AI message replies, bulk message scheduling.
*   **0.5.1:** Import/Export config.
*   **0.5.0:** Keyboard click action, AI image recognition.