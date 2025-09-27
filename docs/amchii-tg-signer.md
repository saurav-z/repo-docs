# tg-signer: Automate Telegram Tasks with Ease 🤖

**Effortlessly automate your Telegram tasks with `tg-signer`, offering features like automated sign-ins, message monitoring, and AI-powered interactions. See the original repo [here](https://github.com/amchii/tg-signer).**

## Key Features

*   **Automated Sign-ins:** Schedule and execute daily Telegram sign-in tasks with flexible timing and random delays.
*   **AI-Powered Interactions:** Utilize AI for image recognition to click buttons and answer calculation questions.
*   **Message Monitoring & Auto-Reply:** Monitor, forward, and automatically respond to messages in your chats, groups, and channels.
*   **Flexible Action Flows:** Configure custom action sequences for tasks like sending text, clicking buttons, and more.
*   **Multi-Account Support:** Run tasks with multiple Telegram accounts simultaneously.
*   **Scheduled Messaging:** Configure Telegram's built-in scheduled message feature.
*   **Message Deletion:** Set messages to be automatically deleted after a specified time.
*   **Advanced Monitoring Rules:** Utilize different rule types, like exact, contains, regex, and all for the monitoring rules.
*   **External Integrations:** Forward messages to external endpoints using UDP and HTTP.

## Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For improved performance, use the speedup option:

```bash
pip install "tg-signer[speedup]"
```

### Docker

You can build a Docker image using the provided `Dockerfile` in the `./docker` directory. See the `./docker/README.md` for instructions.

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

```bash
tg-signer run
tg-signer run my_sign  # Runs the 'my_sign' task directly.
tg-signer run-once my_sign  # Executes the 'my_sign' task once, even if already run today.
tg-signer send-text 8671234001 /test  # Send '/test' to chat ID '8671234001'.
tg-signer send-text -- -10006758812 浇水  # Uses POSIX style for negative chat IDs.
tg-signer send-text --delete-after 1 8671234001 /test  # Sends '/test' and deletes it after 1 second.
tg-signer list-members --chat_id -1001680975844 --admin  # Lists admins of a channel.
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages.
tg-signer monitor run  # Configure message monitoring and auto-reply.
tg-signer multi-run -a account_a -a account_b same_task # Run 'same_task' with multiple accounts.
```

### Configure Proxy (If Needed)

Configure the proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to log in with your phone number and verification code. This will retrieve your recent chat list, which is needed for chat IDs.

### Send a Message

```bash
tg-signer send-text 8671234001 hello  # Sends 'hello' to chat ID '8671234001'.
```

### Run a Sign-in Task

```bash
tg-signer run
```

Or specify the task name:

```bash
tg-signer run linuxdo
```

Configure the tasks through the prompts.

### Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Configure the monitoring via the prompts.

## Data Storage

Configuration and data are stored in the `.signer` directory. You can explore its contents with `tree .signer`:

```
.signer
├── latest_chats.json  # Recent chats
├── me.json  # User info
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitor task name
│       └── config.json  # Monitor configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in record

3 directories, 4 files
```

## Version Changelog

(The version log from the original README)
```
#### 0.8.0
- 支持单个账号同一进程内同时运行多个任务

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