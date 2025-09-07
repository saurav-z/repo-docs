## Automate Your Telegram Tasks with tg-signer

Tired of manually signing in, monitoring messages, and responding in Telegram?  **[tg-signer](https://github.com/amchii/tg-signer)** is your all-in-one solution for automating these tedious tasks.

### Key Features:

*   **Automated Sign-ins:** Schedule daily sign-ins with customizable time offsets.
*   **Keyboard Automation:** Interact with Telegram by clicking buttons based on configured text or image recognition with AI.
*   **Message Monitoring & Response:**  Receive notifications, forward messages, and automatically respond to messages in individual chats, groups, and channels.
*   **Configurable Action Flows:** Create and execute custom action sequences for advanced automation.
*   **Advanced Monitoring Capabilities:** Define complex message matching rules with options for AI-powered responses, forwarding, and notifications.
*   **Schedule Messages:** Leverage Telegram's built-in message scheduling.
*   **Multi-Account Support:** Run tasks with multiple Telegram accounts.

### Installation

Requires Python 3.9 or higher. Install using pip:

```bash
pip install -U tg-signer
```

For faster performance:

```bash
pip install "tg-signer[speedup]"
```

#### Docker

Build your own Docker image using the `Dockerfile` and associated `README` located in the [docker](./docker) directory.

### Usage

Use the following commands to get started:

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
tg-signer run
tg-signer run my_sign  # Run 'my_sign' task without prompting
tg-signer run-once my_sign  # Run 'my_sign' task once, even if already executed today
tg-signer send-text 8671234001 /test  # Send '/test' to chat ID 8671234001
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages
tg-signer monitor run  # Configure and run message monitoring
tg-signer multi-run -a account_a -a account_b same_task  # Run a task with multiple accounts
```

### Configuration

*   **Proxy:** Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option.
    ```bash
    export TG_PROXY=socks5://127.0.0.1:7890
    ```
*   **Login:** Use the `tg-signer login` command to log in and obtain your session. Follow the prompts to enter your phone number and verification code.
*   **Send Text:** Send a message using:
    ```bash
    tg-signer send-text 8671234001 hello
    ```
*   **Run Sign-in Tasks:**
    ```bash
    tg-signer run
    ```
    Or:
    ```bash
    tg-signer run linuxdo
    ```

### Configuration and Storage Location

Configurations and data are saved in the `.signer` directory by default.

### Version Changelog

*(Shortened for brevity)*

*   **0.7.6** Fixes for monitoring multiple chats
*   **0.7.5** RPC error handling and library updates.
*   **0.7.4** Timed actions and cron-based scheduling improvements.
*   **0.7.2** Forward messages to external endpoints (UDP, HTTP).
*   **0.7.0** Multi-action sequences, including text, dice, keyboard clicks, and image selection.
*   **0.6.6** Added dice message support.
*   **0.6.5** Fixed multi-account sign-in record issues.
*   **0.6.4** Simple calculation support and sign-in improvements.
*   **0.6.3** Compatibility updates.
*   **0.6.2** Improved handling of sign-in failures.
*   **0.6.1** Added image recognition after button click.
*   **0.6.0** Cron-based scheduling, all-matching monitor rules, Server酱 push, and multi-run functionality.
*   **0.5.2** Added AI response for monitoring.
*   **0.5.1** `import` and `export` commands.
*   **0.5.0** Keyboard and AI-based image recognition.