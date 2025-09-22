# tg-signer: Automate Telegram Tasks with Python

🤖 Automate your Telegram activities with tg-signer, a versatile tool for signing in, monitoring messages, and interacting with chats.  [View the GitHub Repository](https://github.com/amchii/tg-signer)

## Key Features

*   ✅ **Automated Sign-In:** Schedule daily sign-ins with customizable time offsets.
*   💬 **Message Monitoring & Auto-Reply:**  Monitor personal chats, groups, and channels; then forward or auto-reply based on configurable rules.
*   ⌨️ **Interactive Automation:**  Click keyboard buttons based on configured text or integrate AI-powered image recognition for automated responses.
*   🔄 **Action Flows:** Execute complex task sequences through configurable action flows, including sending text, selecting options, and more.
*   🔄 **Scheduled Messaging:** Leverage Telegram's built-in scheduled messaging feature via the `schedule-messages` command.
*   ⚙️ **Flexible Configuration:**  Utilize environment variables or command-line arguments for proxy, session management, and account configuration.
*   🚀 **Efficient Operation:**  Optional speedup using the `tg-signer[speedup]` extra.
*   🐳 **Docker Support:** Easily deploy using Docker for streamlined setup.

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

Build your own image using the Dockerfile in the [docker](./docker) directory.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  使用<子命令> --help查看使用说明

子命令别名:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  日志等级, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 日志文件路径, 可以是相对路径  [default: tg-signer.log]
  -p, --proxy TEXT                代理地址, 例如: socks5://127.0.0.1:1080,
                                  会覆盖环境变量`TG_PROXY`的值  [env var: TG_PROXY]
  --session_dir PATH              存储TG Sessions的目录, 可以是相对路径  [default: .]
  -a, --account TEXT              自定义账号名称，对应session文件名为<account>.session  [env
                                  var: TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer工作目录，用于存储配置和签到记录等  [default:
                                  .signer]
  --session-string TEXT           Telegram Session String,
                                  会覆盖环境变量`TG_SESSION_STRING`的值  [env var:
                                  TG_SESSION_STRING]
  --in-memory                     是否将session存储在内存中，默认为False，存储在文件
  --help                          Show this message and exit.

Commands:
  export                  导出配置，默认为输出到终端。
  import                  导入配置，默认为从终端读取。
  list                    列出已有配置
  list-members            查询聊天（群或频道）的成员, 频道需要管理员权限
  list-schedule-messages  显示已配置的定时消息
  login                   登录账号（用于获取session）
  logout                  登出账号并删除session文件
  monitor                 配置和运行监控
  multi-run               使用一套配置同时运行多个账号
  reconfig                重新配置
  run                     根据任务配置运行签到
  run-once                运行一次签到任务，即使该签到任务今日已执行过
  schedule-messages       批量配置Telegram自带的定时发送消息功能
  send-text               发送一次消息, 请确保当前会话已经"见过"该`chat_id`
  version                 Show version
```

### Examples

```bash
# Run a sign-in task
tg-signer run

# Run a specific sign-in task directly
tg-signer run my_sign

# Run a sign-in task once, regardless of previous execution
tg-signer run-once my_sign

# Send a text message to a chat
tg-signer send-text 8671234001 /test

# Monitor messages and auto-reply
tg-signer monitor run
```

### Configuring a Proxy (if needed)

Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Logging In

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in.

### Sending a Text Message

```bash
tg-signer send-text 8671234001 hello
```

### Running Sign-In Tasks

```bash
tg-signer run
```

Follow the prompts to configure your sign-in tasks.

### Configuring and Running Message Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your message monitoring rules.

## Configuration & Data Storage

Configuration files and data are stored in the `.signer` directory:

```
.signer
├── latest_chats.json  # Recent Chats
├── me.json  # Personal Information
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitoring task name
│       └── config.json  # Monitoring configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in records