# tg-signer: Automate Telegram Tasks with Ease

**Automate your Telegram tasks with tg-signer, a powerful Python tool for daily check-ins, message monitoring, and automated replies.**  Check out the [original repo](https://github.com/amchii/tg-signer) for the latest updates and source code.

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with customizable time offsets and random delays.
*   **Keyboard Interaction:** Automate actions with text-based keyboard clicks and AI-powered image recognition.
*   **Message Monitoring & Automation:** Monitor, forward, and auto-reply to messages in personal chats, groups, and channels.
*   **Action Flow:** Execute complex task sequences based on your configurations.
*   **Multi-Account Support:** Run multiple Telegram accounts simultaneously.
*   **Scheduled Messages:** Leverage Telegram's built-in scheduled message feature.
*   **Flexible Configuration:** Easily configure tasks via command-line interface.

## Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For speed improvements:

```bash
pip install "tg-signer[speedup]"
```

### Docker

While a pre-built image isn't available, you can build your own using the provided Dockerfile and README in the [docker](./docker) directory.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <子命令> --help for usage instructions

  Aliases:
      run_once -> run-once
      send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level: `debug`, `info`, `warn`, `error`. [default: info]
  --log-file PATH                 Log file path (relative paths supported). [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address (e.g., socks5://127.0.0.1:1080). Overrides TG_PROXY.  [env var: TG_PROXY]
  --session_dir PATH              Directory to store Telegram sessions (relative paths supported).  [default: .]
  -a, --account TEXT              Custom account name, session file named <account>.session.  [env var: TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, storing configurations and check-in records.  [default: .signer]
  --session-string TEXT           Telegram Session String, overrides TG_SESSION_STRING.  [env var: TG_SESSION_STRING]
  --in-memory                     Store session in memory (default: False, stores in files)
  --help                          Show this message and exit.

Commands:
  export                  Export configuration to terminal (default).
  import                  Import configuration from terminal (default).
  list                    List existing configurations.
  list-members            Query chat (group or channel) members (channel requires admin).
  list-schedule-messages  Display configured scheduled messages.
  login                   Log in to your account (to get session).
  logout                  Log out and delete the session file.
  monitor                 Configure and run monitoring.
  multi-run               Run multiple accounts with the same configuration.
  reconfig                Reconfigure.
  run                     Run check-in based on task configuration.
  run-once                Run a check-in task once, even if already executed today.
  schedule-messages       Configure Telegram's built-in scheduled message function.
  send-text               Send a message (ensure chat has "seen" the `chat_id`).
  version                 Show version
```

**Examples:**

```bash
tg-signer run  # Run a configured check-in task
tg-signer run my_sign  # Run the 'my_sign' task directly
tg-signer run-once my_sign  # Run 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat ID '8671234001'
tg-signer send-text -- -10006758812 浇水 # Send to a negative chat ID (use '--' for POSIX style)
tg-signer send-text --delete-after 1 8671234001 /test  # Send '/test' and delete after 1 sec
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好 # Schedule messages
tg-signer monitor run  # Configure message monitoring and auto-replies
tg-signer multi-run -a account_a -a account_b same_task  # Run same_task for multiple accounts
```

## Configuring a Proxy (if needed)

`tg-signer` doesn't use system proxies directly. Configure via the `TG_PROXY` environment variable or the `--proxy` command-line option:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

## Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. This process retrieves your session data.  Ensure the chat you want to interact with is visible in your recent chats.

## Sending a Message

```bash
tg-signer send-text 8671234001 hello  # Send 'hello' to chat ID '8671234001'
```

## Running Check-in Tasks

```bash
tg-signer run
```

Or, specify the task name:

```bash
tg-signer run linuxdo
```

Follow the interactive prompts to configure the task.

### Check-in Task Example:

```
... (configuration prompts) ...
```

## Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the interactive prompts.

### Monitoring Example:

```
... (configuration prompts) ...
```

## Configuration & Data Storage

Configuration data is stored by default in the `.signer` directory.  The following structure exists:

```
.signer
├── latest_chats.json  # Recent chats
├── me.json  # Personal info
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitor task name
│   │   └── config.json  # Monitor config
└── signs  # Check-in tasks
    └── linuxdo  # Check-in task name
        ├── config.json  # Check-in config
        └── sign_record.json  # Check-in records

3 directories, 4 files
```

## Version Changelog

```markdown
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