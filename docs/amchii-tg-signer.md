# TG Signer: Automate Telegram Tasks with Python

**Automate your Telegram interactions with TG Signer, a powerful Python tool for automated sign-ins, message monitoring, and intelligent responses.**  [View the GitHub Repository](https://github.com/amchii/tg-signer)

## Key Features:

*   **Automated Sign-Ins:** Schedule daily sign-ins with customizable timing and randomness.
*   **Keyboard Interaction:**  Configure the tool to click buttons based on configured text or use AI-powered image recognition.
*   **Message Monitoring & Response:**  Monitor personal chats, groups, and channels; forward messages and automatically respond based on flexible rules.
*   **Action Flows:**  Define sequences of actions to execute, including sending text, clicking buttons, image recognition, and math question answering.
*   **AI Integration:** Leverage AI for image recognition and solving calculation questions.
*   **Flexible Configuration:** Easily configure tasks, including chat IDs, keywords, and AI settings.
*   **Advanced Rule Matching:** Utilize exact, contains, regex, and all matching for effective message filtering.
*   **Multi-Account Support:** Run multiple accounts simultaneously with shared or unique configurations.
*   **Message Scheduling:** Schedule Telegram's built-in message sending functionality.
*   **Message Forwarding:**  Forward messages to external endpoints (UDP, HTTP).

## Installation

**Prerequisites:**

*   Python 3.9 or higher

**Install using pip:**

```bash
pip install -U tg-signer
```

**For performance improvements:**

```bash
pip install "tg-signer[speedup]"
```

### Docker

Refer to the [docker](./docker) directory for Dockerfile and instructions.

## Usage

```bash
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <subcommand> --help for usage instructions

Subcommand Aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, which can be a relative path
                                  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, e.g.: socks5://127.0.0.1:1080,
                                  will override the environment variable
                                  `TG_PROXY` [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, which can
                                  be a relative path  [default: .]
  -a, --account TEXT              Custom account name, corresponding session
                                  file name is <account>.session  [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configurations and sign-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String, will override the
                                  environment variable `TG_SESSION_STRING`
                                  [env var: TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory, the
                                  default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to the
                          terminal.
  import                  Import configuration, defaults to read from the
                          terminal.
  list                    List existing configurations
  list-members            Query the members of the chat (group or channel),
                          channels require administrator permissions
  list-schedule-messages  Display scheduled messages
  login                   Login to the account (used to get the session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts with one set of configurations
  reconfig                Reconfigure
  run                     Run the sign-in according to the task configuration
  run-once                Run a sign-in task once, even if the sign-in task
                          has been executed today
  schedule-messages       Batch configure Telegram's built-in scheduled message
                          sending function
  send-text               Send a message once, please make sure that the
                          current session has "seen" the `chat_id`
  version                 Show version
```

**Examples:**

```bash
# Run a sign-in task
tg-signer run

# Run a specific sign-in task
tg-signer run my_sign

# Run a sign-in task once, regardless of previous execution
tg-signer run-once my_sign

# Send a text message
tg-signer send-text 8671234001 /test

# List channel administrators
tg-signer list-members --chat_id -1001680975844 --admin

# Schedule messages
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 Hello

# Configure and run a message monitor
tg-signer monitor run
```

## Configuration

### Proxy Configuration

Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line argument.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. This will retrieve recent chats to ensure your target chats are in the list.

### Sending a Single Message

```bash
tg-signer send-text 8671234001 hello
```

### Running a Sign-in Task

```bash
tg-signer run
```

Or, to run a named task:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

#### Sign-in Task Example:

```
# Example Sign-in Task Configuration
# ... (Configuration steps)
# ...

# Configuration Output
# Chat ID: 7661096533
# Name: jerry bot
# Delete After: 10
# Actions Flow:
# 1. [Send Text] Text: checkin
# 2. [Click Keyboard Button] Click: 签到
# 3. [Select Option based on Image]
# 4. [Reply to Calculation Question]
# 5. [Send Dice Emoji] Dice: 🎲
# ...
# Continue configuring? (y/N): n
# Daily sign-in time (time or crontab expression, such as '06:00:00' or '0 6 * * *'):
# Sign-in time error random seconds (default is 0): 300
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts.

#### Monitor Example:

```
# Example Monitor Configuration
# ... (Configuration steps)
# Chat ID: -4573702599
# Match Rule: contains
# Rule Value: kfc
# Match Users: @neo
# Default Text: V Me 50
# ...

# Chat ID: -4573702599
# Match Rule: regex
# Rule Value: 参与关键词：「.*?」
# Match Users: 61244351
# Extract Text Regex: 参与关键词：「(?P<keyword>(.*?))」\n
# Delete After: 5
# ...

# Chat ID: -4573702599
# Match Rule: all
# Ignore Self: y
# Default Text:
# Reply with AI: n
# ...
# Forward to UDP: y
# UDP Server Address: 127.0.0.1:9999
# ...
# Forward to HTTP: y
# HTTP Address: http://127.0.0.1:8000/tg/user1/messages
```

#### Monitor Example Explanations:

1.  **Chat ID & Usernames:** Supports both integer IDs and usernames (e.g., `@neo`). Note that usernames might not exist.

2.  **Matching Rules:**

    *   `exact`: Exact match.
    *   `contains`:  Contains the specified text (case-insensitive).
    *   `regex`: Matches a regular expression (case-insensitive).  Refer to the Python regex documentation: [Python Regular Expression](https://docs.python.org/zh-cn/3/library/re.html)
    *   `all`: Matches all messages.

3.  **Message Structure**

    ```json
    # Example Message Structure (Partial)
    {
        "_": "Message",
        "id": 2950,
        "from_user": {
            "_": "User",
            "id": 123456789,
            "username": "linuxdo",
            "first_name": "linux",
            "status": "UserStatus.ONLINE",
        },
        "text": "test, 测试",
    }
    ```

#### Monitor Example Output:

```
[INFO] [tg-signer] 2024-10-25 12:29:06,516 core.py 458 开始监控...
[INFO] [tg-signer] 2024-10-25 12:29:37,034 core.py 439 匹配到监控项：MatchConfig(chat_id=-4573702599, rule=contains, rule_value=kfc), default_send_text=V me 50, send_text_search_regex=None
[INFO] [tg-signer] 2024-10-25 12:29:37,035 core.py 442 发送文本：V me 50
[INFO] [tg-signer] 2024-10-25 12:30:02,726 core.py 439 匹配到监控项：MatchConfig(chat_id=-4573702599, rule=regex, rule_value=参与关键词：「.*?」), default_send_text=None, send_text_search_regex=参与关键词：「(?P<keyword>(.*?))」\n
[INFO] [tg-signer] 2024-10-25 12:30:02,727 core.py 442 发送文本：我要抽奖
[INFO] [tg-signer] 2024-10-25 12:30:03,001 core.py 226 Message「我要抽奖」 to -4573702599 will be deleted after 5 seconds.
[INFO] [tg-signer] 2024-10-25 12:30:03,001 core.py 229 Waiting...
[INFO] [tg-signer] 2024-10-25 12:30:08,260 core.py 232 Message「我要抽奖」 to -4573702599 deleted!
```

## Configuration and Data Storage

Configurations and data are stored in the `.signer` directory.

```
.signer
├── latest_chats.json  # Recently retrieved chats
├── me.json  # User Information
├── monitors  # Monitors
│   ├── my_monitor  # Monitor Task Name
│   │   └── config.json  # Monitor Configuration
└── signs  # Sign-in Tasks
    └── linuxdo  # Sign-in Task Name
        ├── config.json  # Sign-in Configuration
        └── sign_record.json  # Sign-in Records
```

## Version Change Log

### 0.7.6
- fix: 监控多个聊天时消息转发至每个聊天 (#55)

### 0.7.5
- 捕获并记录执行任务期间的所有RPC错误
- bump kurigram version to 2.2.7

### 0.7.4
- 执行多个action时，支持固定时间间隔
- 通过`crontab`配置定时执行时不再限制每日执行一次

### 0.7.2
- 支持将消息转发至外部端点，通过：
  - UDP
  - HTTP
- 将kurirogram替换为kurigram

### 0.7.0
- 支持每个聊天会话按序执行多个动作，动作类型：
  - 发送文本
  - 发送骰子
  - 按文本点击键盘
  - 通过图片选择选项
  - 通过计算题回复

### 0.6.6
- 增加对发送DICE消息的支持

### 0.6.5
- 修复使用同一套配置运行多个账号时签到记录共用的问题

### 0.6.4
- 增加对简单计算题的支持
- 改进签到配置和消息处理

### 0.6.3
- 兼容kurigram 2.1.38版本的破坏性变更
> Remove coroutine param from run method [a7afa32](https://github.com/KurimuzonAkuma/pyrogram/commit/a7afa32df208333eecdf298b2696a2da507bde95)

### 0.6.2
- 忽略签到时发送消息失败的聊天

### 0.6.1
- 支持点击按钮文本后继续进行图片识别

### 0.6.0
- Signer支持通过crontab定时
- Monitor匹配规则添加`all`支持所有消息
- Monitor支持匹配到消息后通过server酱推送
- Signer新增`multi-run`用于使用一套配置同时运行多个账号

### 0.5.2
- Monitor支持配置AI进行消息回复
- 增加批量配置「Telegram自带的定时发送消息功能」的功能

### 0.5.1
- 添加`import`和`export`命令用于导入导出配置

### 0.5.0
- 根据配置的文本点击键盘
- 调用AI识别图片点击键盘