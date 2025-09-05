# tg-signer: Automate Your Telegram Tasks with Python

Automate daily check-ins, monitor messages, and create automated responses and more with `tg-signer`, a versatile Python-based tool for Telegram. [View the original repository](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Check-ins:** Schedule and automate daily check-ins with configurable time offsets.
*   **Interactive Actions:** Click buttons based on configured text or utilize AI-powered image recognition for keyboard interactions.
*   **Message Monitoring:** Monitor, forward, and automatically respond to messages in personal chats, groups, and channels.
*   **Configurable Action Flows:** Execute custom action flows based on your configuration.
*   **Flexible Deployment:** Supports installation via pip, and Docker (see [docker](./docker) directory).
*   **Advanced Automation:** Supports sending dice emojis, solving calculation problems, and AI-powered message replies.
*   **Message Forwarding:** Forward messages via UDP and HTTP.
*   **Multi-Account Support:** Easily manage multiple Telegram accounts and run tasks concurrently.
*   **Built-in Scheduling:** Schedule messages using cron expressions.

## Installation

Requires Python 3.9 or higher.

**Using pip:**

```bash
pip install -U tg-signer
```

**For enhanced performance:**

```bash
pip install "tg-signer[speedup]"
```

**Docker (Build your own image):**

Refer to the `docker` directory for the `Dockerfile` and [README](./docker/README.md).

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
  -p, --proxy TEXT                Proxy address, for example: socks5://127.0.0.1:1080,
                                  will override the environment variable
                                  `TG_PROXY`  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path  [default: .]
  -a, --account TEXT              Custom account name, corresponding to the
                                  session file name <account>.session  [env
                                  var: TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and check-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String,
                                  will override the environment variable
                                  `TG_SESSION_STRING`  [env var:
                                  TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to the terminal.
  import                  Import configuration, defaults to reading from the terminal.
  list                    List existing configurations
  list-members            Query members of a chat (group or channel), the channel
                          requires administrator privileges
  list-schedule-messages  Display scheduled messages
  login                   Login to account (used to obtain session)
  logout                  Logout account and delete session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts with a set of configurations at the same time
  reconfig                Reconfigure
  run                     Run check-in based on task configuration
  run-once                Run a check-in task once, even if the check-in task has
                          been executed today
  schedule-messages       Batch configure Telegram's built-in timed message sending
                          function
  send-text               Send a message once, make sure the current session
                          has "seen" the `chat_id`
  version                 Show version
```

**Examples:**

```bash
tg-signer run
tg-signer run my_sign  # Run 'my_sign' task directly, without prompting
tg-signer run-once my_sign  # Run 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id 8671234001
tg-signer send-text -- -10006758812 浇水  # Use '--' before negative chat_id
tg-signer send-text --delete-after 1 8671234001 /test  # Send '/test' and delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List admins of a channel
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages
tg-signer monitor run  # Configure message monitoring and auto-reply
tg-signer multi-run -a account_a -a account_b same_task  # Run same_task config with account_a and account_b
```

## Configuration

### Proxy Configuration

`tg-signer` does not read system proxy settings. Configure proxies using the `TG_PROXY` environment variable or the `--proxy` command-line option.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to log in with your phone number and verification code.

### Sending a Message

```bash
tg-signer send-text 8671234001 hello
```

### Running a Check-in Task

```bash
tg-signer run
```

or specify the task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the check-in.

**Example Check-in Configuration:**

```
...
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to set up monitoring rules.

**Example Monitoring Configuration:**

```
...
```

**Example Explained:**

1.  `chat id` and `user id` support both integer IDs and usernames (usernames must start with `@`).

2.  Matching rules (case-insensitive):

    *   `exact`: Exact message match.
    *   `contains`: Message contains the specified text (e.g., `contains="kfc"`).
    *   `regex`: Regular expression matching (see Python's `re` module).
    *   You can filter messages by specific user IDs.
    *   You can set a default reply text.
    *   You can extract text from messages using regex.

3.  Message Structure (Example):

    ```json
    ...
    ```

**Example Run Output:**

```
...
```

## Version Changelog

```
#### 0.7.6
- fix: 监控多个聊天时消息转发至每个聊天 (#55)

#### 0.7.5
- 捕获并记录执行任务期间的所有RPC错误
- bump kurigram version to 2.2.7

#### 0.7.4
- 执行多个action时，支持固定时间间隔
- 通过`crontab`配置定时执行时不再限制每日执行一次

#### 0.7.2
- 支持将消息转发至外部端点，通过：
  - UDP
  - HTTP
- 将kurirogram替换为kurigram

#### 0.7.0
- 支持每个聊天会话按序执行多个动作，动作类型：
  - 发送文本
  - 发送骰子
  - 按文本点击键盘
  - 通过图片选择选项
  - 通过计算题回复

#### 0.6.6
- 增加对发送DICE消息的支持

#### 0.6.5
- 修复使用同一套配置运行多个账号时签到记录共用的问题

#### 0.6.4
- 增加对简单计算题的支持
- 改进签到配置和消息处理

#### 0.6.3
- 兼容kurigram 2.1.38版本的破坏性变更
> Remove coroutine param from run method [a7afa32](https://github.com/KurimuzonAkuma/pyrogram/commit/a7afa32df208333eecdf298b2696a2da507bde95)


#### 0.6.2
- 忽略签到时发送消息失败的聊天

#### 0.6.1
- 支持点击按钮文本后继续进行图片识别

#### 0.6.0
- Signer支持通过crontab定时
- Monitor匹配规则添加`all`支持所有消息
- Monitor支持匹配到消息后通过server酱推送
- Signer新增`multi-run`用于使用一套配置同时运行多个账号

#### 0.5.2
- Monitor支持配置AI进行消息回复
- 增加批量配置「Telegram自带的定时发送消息功能」的功能

#### 0.5.1
- 添加`import`和`export`命令用于导入导出配置

#### 0.5.0
- 根据配置的文本点击键盘
- 调用AI识别图片点击键盘
```

## Configuration and Data Storage

Data and configurations are stored in the `.signer` directory.

```
.signer
├── latest_chats.json  # 获取的最近对话
├── me.json  # 个人信息
├── monitors  # 监控
│   ├── my_monitor  # 监控任务名
│       └── config.json  # 监控配置
└── signs  # 签到任务
    └── linuxdo  # 签到任务名
        ├── config.json  # 签到配置
        └── sign_record.json  # 签到记录

3 directories, 4 files