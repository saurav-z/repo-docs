# tg-signer: Automate Telegram Tasks - Sign-ins, Monitoring, and More!

Tired of manual Telegram tasks? **Automate your daily Telegram sign-ins, monitor messages, and trigger automated responses with tg-signer!**  [View the original repository here](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Sign-ins:** Schedule and execute daily sign-ins with random delays.
*   **Keyboard Automation:** Click buttons based on configured text or even using AI-powered image recognition.
*   **Message Monitoring & Auto-Response:** Monitor personal chats, groups, and channels; forward messages and set up auto-replies.
*   **Action Flows:** Execute a sequence of actions based on your configuration.
*   **Flexible Configuration:** Supports cron jobs for scheduling and AI integration for advanced interaction.

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

While a pre-built Docker image isn't available, you can easily build your own using the provided `Dockerfile` and associated `README` in the `docker` directory.

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

Here are some example commands:

*   `tg-signer run`: Run all configured sign-in tasks.
*   `tg-signer run my_sign`: Run the sign-in task named 'my_sign'.
*   `tg-signer run-once my_sign`: Run the 'my_sign' task once, even if already executed today.
*   `tg-signer send-text 8671234001 /test`: Send the text '/test' to the chat with ID '8671234001'.
*   `tg-signer send-text -- -10006758812 浇水`: Send text to chat using POSIX style for negative numbers.
*   `tg-signer send-text --delete-after 1 8671234001 /test`: Send a message and delete it after 1 second.
*   `tg-signer list-members --chat_id -1001680975844 --admin`: List channel admins.
*   `tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好`: Schedule messages to be sent.
*   `tg-signer monitor run`: Set up and run message monitoring with auto-reply.
*   `tg-signer multi-run -a account_a -a account_b same_task`: Run a task using multiple accounts.

### Configuring a Proxy (if needed)

`tg-signer` doesn't automatically use system proxies. Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in.  This will also fetch your recent chats, ensuring the desired chat is available for use.

### Sending a Message

```bash
tg-signer send-text 8671234001 hello
```

### Running a Sign-in Task

```bash
tg-signer run
```

Or specify a task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

#### Example Sign-in Configuration:

```
... (Configuration prompts as shown in the original README) ...
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your monitoring rules.

#### Example Monitoring Configuration:

```
... (Configuration prompts and explanations as shown in the original README) ...
```

### Version Changelog

(See the original README for detailed version history.)

### Configuration and Data Storage

Configuration and data are stored in the `.signer` directory.  You can view the directory structure with `tree .signer`:

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