# tg-signer: Automate Telegram Tasks for Daily Sign-ins and More

**Automate your Telegram tasks with `tg-signer`, a versatile tool for daily sign-ins, message monitoring, and automated responses.  [View the original repository](https://github.com/amchii/tg-signer)**

## Key Features

*   **Automated Sign-in:** Schedule daily sign-ins with customizable time windows.
*   **Keyboard Interaction:** Automate actions by clicking keyboard buttons based on text.
*   **AI-Powered Image Recognition:** Use AI to recognize images and interact with buttons.
*   **Message Monitoring & Auto-Response:** Monitor, forward, and automatically reply to messages in personal chats, groups, and channels.
*   **Configurable Action Flows:** Create custom action sequences for complex tasks.
*   **Scheduled Messaging:** Configure and manage Telegram's built-in scheduled messages.

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

*   Build your own Docker image using the [docker](./docker) directory's `Dockerfile` and [README](./docker/README.md).

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
tg-signer run my_sign  # Run the 'my_sign' task without prompting
tg-signer run-once my_sign  # Run the 'my_sign' task once, regardless of previous execution
tg-signer send-text 8671234001 /test  # Send '/test' to chat ID '8671234001'
tg-signer send-text -- -10006758812 浇水  # Use '--' for negative chat IDs
tg-signer send-text --delete-after 1 8671234001 /test  # Send '/test', then delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule a message to be sent for the next 10 days.
tg-signer monitor run  # Configure and run message monitoring and auto-replies
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' config with multiple accounts
```

## Configure Proxy (if needed)

`tg-signer` doesn't use the system proxy by default. Configure proxy settings using the `TG_PROXY` environment variable or the `--proxy` command-line option.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

## Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in. This will fetch your recent chats.

## Send a Single Message

```bash
tg-signer send-text 8671234001 hello
```

## Run Sign-in Tasks

```bash
tg-signer run
```

or specify the task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

### Example Sign-in Configuration

```
# [Sign-in configuration is shown here]
```

## Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure monitoring rules.

### Example Monitoring Configuration

```
# [Monitoring configuration example is shown here]
```

### Detailed Explanation of the Monitoring Configuration

1.  **Chat ID and Usernames:** Both integer IDs and usernames (prefixed with `@`) are supported for specifying chats and users, respectively.
2.  **Matching Rules:**

    *   `exact`: Matches messages that are exactly equal to the specified value.
    *   `contains`: Matches messages that contain the specified value (case-insensitive).
    *   `regex`: Matches messages using a regular expression.  (Refer to [Python Regular Expressions](https://docs.python.org/zh-cn/3/library/re.html)).
    *   `all`: Matches all messages
3.  **Message Structure**

```json
# [JSON message structure is shown here]
```

### Example Run Output

```
# [Example run output is shown here]
```

## Version Change Log

```
# [Version log is shown here]
```

## Configuration and Data Storage

Configuration and data are stored in the `.signer` directory.

```
.signer
├── latest_chats.json  # Recent conversations
├── me.json  # Personal information
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Task name
│       └── config.json  # Monitor configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Task name
        ├── config.json  # Sign-in config
        └── sign_record.json  # Sign-in records

3 directories, 4 files