# tg-signer: Automate Your Telegram Tasks with Python

**Automate Telegram tasks like daily check-ins, message monitoring, and auto-replies with this powerful Python tool. [View the GitHub repository](https://github.com/amchii/tg-signer).**

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with customizable time and random delays.
*   **Interactive Automation:** Click buttons based on configured text or use AI-powered image recognition to click on specific elements.
*   **Message Monitoring and Handling:** Monitor personal chats, groups, and channels, with options for forwarding and auto-replies.
*   **Configurable Action Flows:** Define and execute action flows for complex automation tasks.
*   **Flexible Configuration:** Configure proxies, account names, and session storage.
*   **Message Scheduling:** Leverage Telegram's built-in message scheduling.
*   **Multi-Account Support:** Run tasks on multiple accounts simultaneously.
*   **Extensible with AI:** Integrate AI models for advanced image recognition and calculation problem solving.

## Installation

Requires Python 3.9 or later.

**Install with pip:**

```bash
pip install -U tg-signer
```

**For faster performance (optional):**

```bash
pip install "tg-signer[speedup]"
```

**Docker (Build your own image):**

See the [docker](./docker) directory for the `Dockerfile` and [README](./docker/README.md).

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

**Examples:**

```bash
tg-signer run  # Run configured check-in tasks
tg-signer run my_sign  # Run a specific task
tg-signer run-once my_sign  # Run a task once, even if already run today
tg-signer send-text 8671234001 /test  # Send a message to a chat
tg-signer monitor run  # Configure and start message monitoring
tg-signer multi-run -a account_a -a account_b same_task  # Run a task on multiple accounts
```

## Configuration

### Proxy Configuration (if needed)

Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in.

### Sending a Single Message

```bash
tg-signer send-text 8671234001 hello
```

### Running Check-in Tasks

```bash
tg-signer run
```

Or run a named task:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.  See the original README for detailed task configuration examples.

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure the monitoring rules.  See the original README for detailed monitoring configuration examples.

## Data Storage

Configuration and data are stored in the `.signer` directory.

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
```

## Changelog

See the original README for the changelog.