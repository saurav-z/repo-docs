# tg-signer: Automate Telegram Tasks with Python

**Automate your Telegram experience with tg-signer, a powerful Python tool for daily check-ins, message monitoring, and automated responses.**  [View the original repository on GitHub](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with configurable time offsets.
*   **Keyboard Interaction:** Interact with Telegram bots by clicking buttons based on configured text.
*   **AI-Powered Image Recognition:** Use AI to identify and interact with elements in images.
*   **Message Monitoring and Auto-Reply:** Monitor personal chats, groups, and channels, with the ability to forward and automatically reply to messages.
*   **Configurable Action Flows:** Execute complex actions based on your configurations, including sending text, clicking buttons, and more.
*   **Scheduled Messaging:** Utilize Telegram's built-in scheduling feature.
*   **Multi-Account Support:** Run multiple Telegram accounts simultaneously.
*   **Flexible Configuration:** Supports environment variables and command-line arguments for easy setup.

## Installation

Requires Python 3.9 or higher.

**Install with pip:**

```bash
pip install -U tg-signer
```

**For enhanced speed:**

```bash
pip install "tg-signer[speedup]"
```

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

*   **Run a check-in:**

```bash
tg-signer run
tg-signer run my_sign  # Run the 'my_sign' task directly (without prompting)
tg-signer run-once my_sign  # Run the 'my_sign' task once, regardless of previous executions
```

*   **Send a text message:**

```bash
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id 8671234001
```

*   **List members of a chat:**

```bash
tg-signer list-members --chat_id -1001680975844 --admin  # List admins of a channel
```

*   **Schedule messages using crontab:**

```bash
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Send "你好" to chat_id '-1001680975844' at 0:00 for 10 days
```

*   **Monitor messages:**

```bash
tg-signer monitor run  # Configure message monitoring and auto-reply
```

*   **Run multiple accounts with same task:**

```bash
tg-signer multi-run -a account_a -a account_b same_task # Run 'same_task' configuration for 'account_a' and 'account_b'
```

### Configuration

*   **Proxy:** Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line argument.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in and fetch your recent chats.

### Data Storage Location

Configuration and data are stored in the `.signer` directory.  For example:

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

## Monitoring Configuration

The `tg-signer monitor run` command is used for configuring and running message monitoring and auto-reply rules. This feature lets you automatically respond to specific messages based on keywords, regular expressions, and user filters.

### Example Configuration Steps:

1.  **Enter Chat ID:** Provide the chat ID (integer or username starting with @).
2.  **Matching Rule:** Choose a matching rule (`exact`, `contains`, `regex`, `all`).
3.  **Rule Value:** Enter the text, regex, or keyword for matching.
4.  **User Filter:** Optionally specify the user(s) to filter messages from (comma-separated user IDs or usernames).  Leave blank to match all users.
5.  **Default Send Text:** Define the text to be sent automatically upon a match.
6.  **Regex Extract Text (Optional):**  Use a regex to extract and send specific text from matching messages.
7.  **Delete after:** wait for certain seconds, then the message will be deleted.
8.  **Continue/End:** Add more items, then the configuration is done.

**For detailed information about the message structure and examples, refer to the complete README on the GitHub repository.**