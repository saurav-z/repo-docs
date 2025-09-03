# tg-signer: Automate Telegram Tasks with Python 🤖

**Automate your Telegram interactions with tg-signer, a powerful Python tool for daily check-ins, message monitoring, and automated responses.**  [View the original repo on GitHub](https://github.com/amchii/tg-signer).

## Key Features:

*   ✅ **Automated Check-ins:** Schedule and automate daily Telegram check-ins with customizable timings and error margins.
*   💬 **Message Monitoring & Response:** Monitor personal chats, groups, and channels, with automated forwarding and replies based on customizable rules.
*   ⌨️ **Keyboard Interactions:** Automatically interact with Telegram's in-app keyboards based on text input or AI-powered image recognition.
*   🖼️ **AI-Powered Actions:** Integrate AI for image recognition and response to calculation questions.
*   🔄 **Flexible Action Flows:** Define complex action flows with multiple steps, including sending text, clicking buttons, and AI-based interactions.
*   🚀 **Multi-Account Support:** Run tasks across multiple Telegram accounts simultaneously.
*   🕰️ **Scheduled Messages:** Configure and manage Telegram's built-in scheduled message feature.
*   🌐 **Proxy Support:** Configure proxy settings using environment variables or command-line options.
*   📦 **Docker Support:** Easily deploy with provided Dockerfile and documentation.

## Installation

**Prerequisites:** Python 3.9 or higher

Install using pip:

```bash
pip install -U tg-signer
```

For improved performance:

```bash
pip install "tg-signer[speedup]"
```

### Docker

Build a Docker image using the provided `Dockerfile` in the `docker` directory (refer to the `docker/README.md` for details).

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

### Examples:

*   **Run a check-in task:**

    ```bash
    tg-signer run
    tg-signer run my_sign  # Run a specific task
    tg-signer run-once my_sign # Run a task once
    ```

*   **Send a text message:**

    ```bash
    tg-signer send-text 8671234001 /test  # Send to a specific chat ID
    ```

*   **List members of a chat:**

    ```bash
    tg-signer list-members --chat_id -1001680975844 --admin  # List admins of a channel
    ```

*   **Configure and run message monitoring:**

    ```bash
    tg-signer monitor run
    ```

*   **Run tasks across multiple accounts:**

    ```bash
    tg-signer multi-run -a account_a -a account_b same_task
    ```

### Configuration

*   **Proxy:** Configure proxy settings using the `TG_PROXY` environment variable or the `--proxy` command-line option.
    ```bash
    export TG_PROXY=socks5://127.0.0.1:7890
    ```
*   **Login:** Use `tg-signer login` to authenticate your Telegram account.
*   **Sign-in Task Configuration:**  `tg-signer run`, follow the prompts to set up the following actions:

    *   Send Text
    *   Send Dice Emoji
    *   Click Keyboard Buttons
    *   Select Options via Image Recognition
    *   Answer Calculation Questions

*   **Monitor Task Configuration:** `tg-signer monitor run my_monitor`, you can set up various triggers for monitoring messages:
    *   `chat id`, `user id`. username must be prefixed with `@`.
    *   Match Rules: `exact`, `contains`, `regex`, `all`
    *   Default Text
    *   Text Extraction from Messages
    *   Ignore messages from self

## Data Storage

Configurations and data are stored in the `.signer` directory:

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

## Version History
See the original repo for version change logs.

---