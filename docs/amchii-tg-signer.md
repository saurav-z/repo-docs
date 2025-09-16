# TG Signer: Automate Telegram Tasks for Daily Sign-ins, Monitoring, and More

**Effortlessly automate your Telegram tasks with TG Signer!** ([Original Repository](https://github.com/amchii/tg-signer))

## Key Features

*   **Automated Sign-in:** Schedule daily sign-in tasks with customizable timing and random delays.
*   **Interactive Keyboard Actions:** Automatically click keyboard buttons based on configured text.
*   **AI-Powered Image Recognition:** Utilize AI to recognize and click keyboard options from images.
*   **Message Monitoring & Automation:** Monitor personal chats, groups, and channels, with options for forwarding and auto-replying.
*   **Flexible Action Flows:** Execute complex actions based on custom configurations.
*   **Schedule Messages:** Easily set up scheduled messages using Telegram's built-in functionality.
*   **Multi-Account Support:** Run tasks across multiple Telegram accounts simultaneously.

## Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For faster performance:

```bash
pip install "tg-signer[speedup]"
```

### Docker

You can build your own Docker image using the provided `Dockerfile` and `README` in the [docker](./docker) directory.

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

Examples:

```bash
tg-signer run
tg-signer run my_sign  # Runs the 'my_sign' task directly, without prompting.
tg-signer run-once my_sign  # Runs the 'my_sign' task once, regardless of previous runs.
tg-signer send-text 8671234001 /test  # Sends '/test' to chat ID '8671234001'.
tg-signer send-text -- -10006758812 浇水  # Use '--' for negative chat IDs with POSIX style.
tg-signer send-text --delete-after 1 8671234001 /test  # Sends '/test' and deletes after 1 second.
tg-signer list-members --chat_id -1001680975844 --admin  # Lists channel admins.
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedules a message.
tg-signer monitor run  # Configures and runs message monitoring with auto-reply.
tg-signer multi-run -a account_a -a account_b same_task  # Runs 'same_task' with multiple accounts.
```

## Configuration

### Proxy Configuration (if needed)

Configure proxies using the `TG_PROXY` environment variable or the `--proxy` command option:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in and retrieve your chat list.

### Send a Single Message

```bash
tg-signer send-text 8671234001 hello
```

### Run a Sign-in Task

```bash
tg-signer run
```

Or, specify a task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure your sign-in task.

### Monitor and Configure Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to set up your monitoring rules.

## Version Changelog

*   **0.7.6**
    *   Fix: Monitoring multiple chats now correctly forwards messages to each chat.
*   **0.7.5**
    *   Captures and logs all RPC errors during task execution.
    *   Bumped kurigram version to 2.2.7
*   **0.7.4**
    *   Added fixed time intervals when executing multiple actions.
    *   Crontab configurations for scheduled execution no longer limit execution to once per day.
*   **0.7.2**
    *   Supports forwarding messages to external endpoints via:
        *   UDP
        *   HTTP
    *   Replaced `kurirogram` with `kurigram`.
*   **0.7.0**
    *   Supports sequential execution of multiple actions per chat session, with action types including:
        *   Send text
        *   Send dice
        *   Click keyboard buttons by text
        *   Select options via image
        *   Respond to math problems
*   **0.6.6**
    *   Added support for sending DICE messages.
*   **0.6.5**
    *   Fixed issue where sign-in records were shared when running multiple accounts with the same configuration.
*   **0.6.4**
    *   Added support for simple math problems.
    *   Improved sign-in configuration and message handling.
*   **0.6.3**
    *   Compatibility fix for breaking changes in kurigram version 2.1.38.
*   **0.6.2**
    *   Ignores chats that fail to send messages during sign-in.
*   **0.6.1**
    *   Supports performing image recognition after clicking a button.
*   **0.6.0**
    *   Signer now supports scheduling via crontab.
    *   Monitor rule added `all` support to match all messages.
    *   Monitor supports pushing messages via server-chan after matching.
    *   Signer added `multi-run` to run multiple accounts with a single configuration.
*   **0.5.2**
    *   Monitor supports AI-powered message replies.
    *   Added a feature to configure Telegram's built-in message scheduling.
*   **0.5.1**
    *   Added `import` and `export` commands for configuration import/export.
*   **0.5.0**
    *   Click keyboard buttons by text.
    *   Use AI to recognize and click keyboard options from images.

## Configuration and Data Storage

Data and configurations are stored in the `.signer` directory. You can see the file structure with `tree .signer`:

```
.signer
├── latest_chats.json  # Recent chats
├── me.json  # Personal information
├── monitors  # Monitors
│   ├── my_monitor  # Monitor task name
│   │   └── config.json  # Monitor configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in records

3 directories, 4 files