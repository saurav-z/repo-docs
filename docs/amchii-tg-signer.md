# tg-signer: Automate Telegram Tasks with Python

**Automate your Telegram experience with `tg-signer`, a powerful Python-based tool for automated daily check-ins, message monitoring, and intelligent responses. [See the original repo](https://github.com/amchii/tg-signer)**

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with customizable time offsets and random delays.
*   **Interactive Keyboard Actions:** Automate interactions by clicking on keyboard buttons based on configured text.
*   **AI-Powered Image Recognition:** Leverage AI to identify and interact with elements in images for automated responses.
*   **Advanced Message Monitoring:**  Monitor personal, group, and channel messages with flexible rule-based responses and forwarding.
*   **Flexible Action Flows:** Define complex task sequences using various action types for advanced automation.
*   **Flexible Scheduling Options:** Support for both time-based and cron-based scheduling for automated task execution.
*   **Multi-Account Support:** Run multiple Telegram accounts simultaneously.
*   **AI Integration:** Replies can be generated via AI using OpenAI.

## Installation

Ensure you have Python 3.9 or higher installed.

```bash
pip install -U tg-signer
```

For performance improvements:

```bash
pip install "tg-signer[speedup]"
```

### Docker

You can build your own Docker image using the provided `Dockerfile` and instructions in the  [docker](./docker) directory's README.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <子命令> --help to view usage instructions

Subcommand aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, can be a relative path  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, e.g.: socks5://127.0.0.1:1080,
                                  will overwrite the environment variable `TG_PROXY`  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a relative path  [default: .]
  -a, --account TEXT              Custom account name, corresponding session filename is <account>.session  [env
                                  var: TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store configuration and check-in records, etc.  [default:
                                  .signer]
  --session-string TEXT           Telegram Session String,
                                  will overwrite the environment variable `TG_SESSION_STRING`  [env var:
                                  TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory, default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, default output to terminal.
  import                  Import configuration, default read from terminal.
  list                    List existing configurations
  list-members            Query members of a chat (group or channel), channel requires admin permissions
  list-schedule-messages  Display configured scheduled messages
  login                   Log in to account (used to get session)
  logout                  Log out of account and delete session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts simultaneously using a set of configurations
  reconfig                Reconfigure
  run                     Run check-in based on task configuration
  run-once                Run a check-in task once, even if the check-in task has been executed today
  schedule-messages       Configure Telegram's built-in scheduled message function in batches
  send-text               Send a message once, please make sure the current session has "seen" the `chat_id`
  version                 Show version
```

## Examples

```bash
# Run a check-in task
tg-signer run

# Run a specific check-in task
tg-signer run my_sign

# Run a check-in task once
tg-signer run-once my_sign

# Send a text message
tg-signer send-text 8671234001 /test

# List channel admins
tg-signer list-members --chat_id -1001680975844 --admin

# Schedule messages with cron
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好

# Configure and run monitoring
tg-signer monitor run

# Run multiple accounts with same config
tg-signer multi-run -a account_a -a account_b same_task
```

## Configuration

### Configure Proxy (if needed)

`tg-signer` does not automatically use system proxies. Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. The login process retrieves your recent chats, ensuring the desired chats are listed.

### Send a Message

```bash
tg-signer send-text 8671234001 hello
```

### Run a Check-in Task

```bash
tg-signer run
```

Or specify a preconfigured task:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

#### Example Check-in Configuration

```
...
第1个签到
一. Chat ID（登录时最近对话输出中的ID）: 7661096533
二. Chat名称（可选）: jerry bot
三. 开始配置<动作>，请按照实际签到顺序配置。
  1: 发送普通文本
  2: 发送Dice类型的emoji
  3: 根据文本点击键盘
  4: 根据图片选择选项
  5: 回复计算题

第1个动作:
1. 输入对应的数字选择动作: 1
2. 输入要发送的文本: checkin
3. 是否继续添加动作？(y/N)：y
第2个动作:
1. 输入对应的数字选择动作: 3
2. 键盘中需要点击的按钮文本: 签到
3. 是否继续添加动作？(y/N)：y
第3个动作:
1. 输入对应的数字选择动作: 4
图片识别将使用大模型回答，请确保大模型支持图片识别。
2. 是否继续添加动作？(y/N)：y
第4个动作:
1. 输入对应的数字选择动作: 5
计算题将使用大模型回答。
2. 是否继续添加动作？(y/N)：y
第5个动作:
1. 输入对应的数字选择动作: 2
2. 输入要发送的骰子（如 🎲, 🎯）: 🎲
3. 是否继续添加动作？(y/N)：n
在运行前请通过环境变量正确设置`OPENAI_API_KEY`, `OPENAI_BASE_URL`。默认模型为"gpt-4o", 可通过环境变量`OPENAI_MODEL`更改。
四. 等待N秒后删除签到消息（发送消息后等待进行删除, '0'表示立即删除, 不需要删除直接回车）, N: 10
╔════════════════════════════════════════════════╗
║ Chat ID: 7661096533                            ║
║ Name: jerry bot                                ║
║ Delete After: 10                               ║
╟────────────────────────────────────────────────╢
║ Actions Flow:                                  ║
║ 1. [发送普通文本] Text: checkin                ║
║ 2. [根据文本点击键盘] Click: 签到              ║
║ 3. [根据图片选择选项]                          ║
║ 4. [回复计算题]                                ║
║ 5. [发送Dice类型的emoji] Dice: 🎲              ║
╚════════════════════════════════════════════════╝
第1个签到配置成功

继续配置签到？(y/N)：n
每日签到时间（time或crontab表达式，如'06:00:00'或'0 6 * * *'）:
签到时间误差随机秒数（默认为0）: 300
```

### Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your monitoring rules.

#### Example Monitoring Configuration

```
...
配置第1个监控项
1. Chat ID（登录时最近对话输出中的ID）: -4573702599
2. 匹配规则('exact', 'contains', 'regex', 'all'): contains
3. 规则值（不可为空）: kfc
4. 只匹配来自特定用户ID的消息（多个用逗号隔开, 匹配所有用户直接回车）: @neo
5. 默认发送文本: V Me 50
6. 从消息中提取发送文本的正则表达式:
7. 等待N秒后删除签到消息（发送消息后等待进行删除, '0'表示立即删除, 不需要删除直接回车）, N:
继续配置？(y/N)：y

配置第2个监控项
1. Chat ID（登录时最近对话输出中的ID）: -4573702599
2. 匹配规则('exact', 'contains', 'regex'): regex
3. 规则值（不可为空）: 参与关键词：「.*?」
4. 只匹配来自特定用户ID的消息（多个用逗号隔开, 匹配所有用户直接回车）: 61244351
5. 默认发送文本:
6. 从消息中提取发送文本的正则表达式: 参与关键词：「(?P<keyword>(.*?))」\n
7. 等待N秒后删除签到消息（发送消息后等待进行删除, '0'表示立即删除, 不需要删除直接回车）, N: 5
继续配置？(y/N)：y

配置第3个监控项
1. Chat ID（登录时最近对话输出中的ID）: -4573702599
2. 匹配规则(exact, contains, regex, all): all
3. 只匹配来自特定用户ID的消息（多个用逗号隔开, 匹配所有用户直接回车）:
4. 总是忽略自己发送的消息（y/N）: y
5. 默认发送文本（不需要则回车）:
6. 是否使用AI进行回复(y/N): n
7. 从消息中提取发送文本的正则表达式（不需要则直接回车）:
8. 是否通过Server酱推送消息(y/N): n
9. 是否需要转发到外部（UDP, Http）(y/N): y
10. 是否需要转发到UDP(y/N): y
11. 请输入UDP服务器地址和端口（形如`127.0.0.1:1234`）: 127.0.0.1:9999
12. 是否需要转发到Http(y/N): y
13. 请输入Http地址（形如`http://127.0.0.1:1234`）: http://127.0.0.1:8000/tg/user1/messages
继续配置？(y/N)：n
```

#### Monitoring Configuration Details

1.  **Chat ID and User ID:**  Both support integer IDs and usernames (prefixed with `@`). Note that usernames might not always exist.
2.  **Matching Rules (case-insensitive):**
    *   `exact`: Exact match required.
    *   `contains`:  Message must contain the specified value.
    *   `regex`:  Regular expression matching (Python regex syntax).
    *   `all`: Matches all messages.
3.  **Message Structure:** Refer to the provided JSON example in the original README for the message structure.

## Version Changelog
... (Changelog content from original README)

## Configuration and Data Storage

Configuration and data are stored in the `.signer` directory.

```bash
tree .signer
```

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