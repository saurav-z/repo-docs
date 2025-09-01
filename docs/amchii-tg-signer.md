# tg-signer: Automate Your Telegram Tasks with Python

**Automate daily check-ins, monitor messages, and set up auto-replies, all within Telegram, with the power of Python. Check out the original repo [here](https://github.com/amchii/tg-signer)!**

## Key Features

*   **Automated Tasks:** Schedule daily check-ins with flexible time settings, including random delays.
*   **Interactive Automation:** Automate actions like clicking keyboard buttons based on text, or even using AI-powered image recognition.
*   **Smart Monitoring & Auto-Reply:** Keep tabs on personal chats, groups, and channels, with the ability to forward and auto-reply to messages based on custom rules.
*   **Customizable Action Flows:** Define and execute complex action sequences to streamline your Telegram interactions.
*   **Flexible Deployment:**  Works with standard Python installations, or boost speed using the `[speedup]` install, or with Docker.

## Installation

Requires Python 3.9 or higher.

Install via pip:

```bash
pip install -U tg-signer
```

For improved performance:

```bash
pip install "tg-signer[speedup]"
```

### Docker

Build your own Docker image using the `Dockerfile` and README in the [docker](./docker) directory.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <子命令> --help for usage instructions.

Subcommand aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, can be a relative path
                                  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, e.g.: socks5://127.0.0.1:1080,
                                  will override the environment variable `TG_PROXY`
                                  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path  [default: .]
  -a, --account TEXT              Custom account name, the corresponding
                                  session file name is <account>.session
                                  [env var: TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and sign-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String,
                                  will override the environment variable
                                  `TG_SESSION_STRING`  [env var:
                                  TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  default is False, stored in file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to terminal.
  import                  Import configuration, defaults to read from terminal.
  list                    List existing configurations
  list-members            Query the members of the chat (group or channel),
                          channels require administrator privileges
  list-schedule-messages  Display configured scheduled messages
  login                   Login to the account (used to obtain the session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts at the same time using a set of
                          configurations
  reconfig                Reconfigure
  run                     Run check-in according to task configuration
  run-once                Run a check-in task once, even if the check-in task
                          has been executed today
  schedule-messages       Configure Telegram's built-in scheduled message
                          function in batches
  send-text               Send a message once, please make sure that the
                          current session has "seen" the `chat_id`
  version                 Show version
```

### Examples

```bash
tg-signer run
tg-signer run my_sign  # Run 'my_sign' task directly without prompting
tg-signer run-once my_sign  # Run 'my_sign' task once immediately
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id '8671234001'
tg-signer send-text -- -10006758812 浇水  # Use '--' for negative chat IDs
tg-signer send-text --delete-after 1 8671234001 /test  # Send and delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages for 10 days at 0:00
tg-signer monitor run  # Configure message monitoring and auto-replies
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' with multiple accounts
```

### Configure Proxy (if needed)

`tg-signer` does not use system proxies; configure with the `TG_PROXY` environment variable or the `--proxy` command parameter.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code to log in and fetch recent chats.

### Send a Message

```bash
tg-signer send-text 8671234001 hello  # Send 'hello' to chat_id '8671234001'
```

### Run a Check-in Task

```bash
tg-signer run
```

Or specify the task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure your task.

#### Example Task Configuration:

```
# ... (Configuration prompts as described in the original README) ...
```

### Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts.

#### Example Monitor Configuration:

```
# ... (Configuration prompts and explanations as described in the original README) ...
```

### Version Changelog

#### 0.7.6
*   fix: 监控多个聊天时消息转发至每个聊天 (#55)

#### 0.7.5
*   捕获并记录执行任务期间的所有RPC错误
*   bump kurigram version to 2.2.7

#### 0.7.4
*   执行多个action时，支持固定时间间隔
*   通过`crontab`配置定时执行时不再限制每日执行一次

#### 0.7.2
*   支持将消息转发至外部端点，通过：
  - UDP
  - HTTP
*   将kurirogram替换为kurigram

#### 0.7.0
*   支持每个聊天会话按序执行多个动作，动作类型：
  - 发送文本
  - 发送骰子
  - 按文本点击键盘
  - 通过图片选择选项
  - 通过计算题回复

#### 0.6.6
*   增加对发送DICE消息的支持

#### 0.6.5
*   修复使用同一套配置运行多个账号时签到记录共用的问题

#### 0.6.4
*   增加对简单计算题的支持
*   改进签到配置和消息处理

#### 0.6.3
*   兼容kurigram 2.1.38版本的破坏性变更
> Remove coroutine param from run method [a7afa32](https://github.com/KurimuzonAkuma/pyrogram/commit/a7afa32df208333eecdf298b2696a2da507bde95)


#### 0.6.2
*   忽略签到时发送消息失败的聊天

#### 0.6.1
*   支持点击按钮文本后继续进行图片识别

#### 0.6.0
*   Signer支持通过crontab定时
*   Monitor匹配规则添加`all`支持所有消息
*   Monitor支持匹配到消息后通过server酱推送
*   Signer新增`multi-run`用于使用一套配置同时运行多个账号

#### 0.5.2
*   Monitor支持配置AI进行消息回复
*   增加批量配置「Telegram自带的定时发送消息功能」的功能

#### 0.5.1
*   添加`import`和`export`命令用于导入导出配置

#### 0.5.0
*   根据配置的文本点击键盘
*   调用AI识别图片点击键盘

### Configuration and Data Storage

Configurations and data are stored in the `.signer` directory.  Use `tree .signer` to view the structure.

```
# ... (tree .signer output as described in the original README) ...