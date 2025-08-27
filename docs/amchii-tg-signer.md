# tg-signer: Automate Your Telegram Tasks with Python

**Automate your Telegram interactions with `tg-signer`, a versatile Python-based tool for automatic check-ins, message monitoring, and intelligent responses.**  [View the project on GitHub](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Check-ins:**
    *   Perform daily check-ins with configurable timings and random delays.
    *   Interact with keyboards based on configured text and image recognition (using AI).
*   **Intelligent Monitoring and Response:**
    *   Monitor personal chats, group chats, and channels.
    *   Forward and automatically reply to messages based on customizable rules.
    *   Utilize AI for image recognition and automated responses.
*   **Flexible Configuration:**
    *   Define action flows for complex tasks.
    *   Schedule tasks using time or cron expressions.
    *   Supports multiple Telegram accounts and simultaneous task execution.
*   **Extensive Command-Line Interface (CLI):**
    *   Manage configurations, accounts, and tasks through intuitive commands.
    *   Configure and run monitoring and automated responses.
    *   Utilize a variety of commands to manage sessions, send messages, and more.

## Installation

`tg-signer` requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For improved performance, install with the speedup option:

```bash
pip install "tg-signer[speedup]"
```

### Docker

You can build your own Docker images using the provided `Dockerfile` and the instructions in the [docker/README.md](./docker/README.md) file within the repository.

## Usage

The CLI provides a wide range of commands:

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <子命令> --help查看使用说明

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

```bash
tg-signer run  # Run a configured check-in task
tg-signer run my_sign  # Run a check-in task named 'my_sign' directly
tg-signer run-once my_sign  # Run a check-in task once, even if already run today
tg-signer send-text 8671234001 /test  # Send '/test' to chat ID 8671234001
tg-signer send-text -- -10006758812 浇水  # Send '浇水' to chat ID -10006758812 (using POSIX style)
tg-signer send-text --delete-after 1 8671234001 /test  # Send '/test' and delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List admins of a channel
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages
tg-signer monitor run  # Configure and run message monitoring and auto-reply
tg-signer multi-run -a account_a -a account_b same_task  # Run same_task with multiple accounts
```

### Configuring Proxies (If Needed)

`tg-signer` does *not* automatically use system proxies. Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line argument.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. This will log you in and retrieve your recent chats.

### Send a Message

```bash
tg-signer send-text 8671234001 hello  # Send 'hello' to chat ID 8671234001
```

### Run a Check-in Task

```bash
tg-signer run
```

Or, run a specific task directly:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

**Example Task Configuration:**

```
开始配置任务<linuxdo>
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

**Example Monitoring Configuration:**

```
开始配置任务<my_monitor>
聊天chat id和用户user id均同时支持整数id和字符串username, username必须以@开头，如@neo

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

**Example Monitoring Explanation:**

1.  Both chat IDs and user IDs support integer IDs and usernames, however usernames *must* start with @. For example, if you wish to only listen to messages from `@neo` the user ID should be `@neo`. *Be aware that the username may not always exist.* In the example, the chat ID of -4573702599 indicates that the rules apply only to the corresponding chat.

2.  Matching rules, which are currently case-insensitive:

    *   `exact`: The message must exactly match the specified value.
    *   `contains`: The message must contain the specified value.  For example, if `contains="kfc"`, any message containing "kfc" (e.g., "I like MacDonalds rather than KfC") will match (case-insensitive).
    *   `regex`: Regular expression matching (see [Python Regular Expressions](https://docs.python.org/zh-cn/3/library/re.html)). If the regular expression is *found* in the message, it will match.  The example "参与关键词：「.*?」" will match the message: "新的抽奖已经创建... 参与关键词：「我要抽奖」 建议先私聊机器人".
    *   You can match messages from specific users; for example, only group administrators instead of any user.
    *   You can set a default text to be sent when a message is matched.
    *   You can extract text from the message using a regular expression. For example, "参与关键词：「(.*?)」\n" with the use of capture groups `(...)` to extract the text. This can capture the keyword "我要抽奖" from the above example, and automatically send it.

3.  Message structure reference:

    ```json
    {
        "_": "Message",
        "id": 2950,
        "from_user": {
            "_": "User",
            "id": 123456789,
            "is_self": false,
            "is_contact": false,
            "is_mutual_contact": false,
            "is_deleted": false,
            "is_bot": false,
            "is_verified": false,
            "is_restricted": false,
            "is_scam": false,
            "is_fake": false,
            "is_support": false,
            "is_premium": false,
            "is_contact_require_premium": false,
            "is_close_friend": false,
            "is_stories_hidden": false,
            "is_stories_unavailable": true,
            "is_business_bot": false,
            "first_name": "linux",
            "status": "UserStatus.ONLINE",
            "next_offline_date": "2025-05-30 11:52:40",
            "username": "linuxdo",
            "dc_id": 5,
            "phone_number": "*********",
            "photo": {
                "_": "ChatPhoto",
                "small_file_id": "AQADBQADqqcxG6hqrTMAEAIAA6hqrTMABLkwVDcqzBjAAAQeBA",
                "small_photo_unique_id": "AgADqqcxG6hqrTM",
                "big_file_id": "AQADBQADqqcxG6hqrTMAEAMAA6hqrTMABLkwVDcqzBjAAAQeBA",
                "big_photo_unique_id": "AgADqqcxG6hqrTM",
                "has_animation": false,
                "is_personal": false
            },
            "added_to_attachment_menu": false,
            "inline_need_location": false,
            "can_be_edited": false,
            "can_be_added_to_attachment_menu": false,
            "can_join_groups": false,
            "can_read_all_group_messages": false,
            "has_main_web_app": false
        },
        "date": "2025-05-30 11:47:46",
        "chat": {
            "_": "Chat",
            "id": -52737131599,
            "type": "ChatType.GROUP",
            "is_creator": true,
            "is_deactivated": false,
            "is_call_active": false,
            "is_call_not_empty": false,
            "title": "测试组",
            "has_protected_content": false,
            "members_count": 4,
            "permissions": {
                "_": "ChatPermissions",
                "can_send_messages": true,
                "can_send_media_messages": true,
                "can_send_other_messages": true,
                "can_send_polls": true,
                "can_add_web_page_previews": true,
                "can_change_info": true,
                "can_invite_users": true,
                "can_pin_messages": true,
                "can_manage_topics": true
            }
        },
        "from_offline": false,
        "show_caption_above_media": false,
        "mentioned": false,
        "scheduled": false,
        "from_scheduled": false,
        "edit_hidden": false,
        "has_protected_content": false,
        "text": "test, 测试",
        "video_processing_pending": false,
        "outgoing": false
    }
    ```

**Example Run Output:**

```
[INFO] [tg-signer] 2024-10-25 12:29:06,516 core.py 458 开始监控...
[INFO] [tg-signer] 2024-10-25 12:29:37,034 core.py 439 匹配到监控项：MatchConfig(chat_id=-4573702599, rule=contains, rule_value=kfc), default_send_text=V me 50, send_text_search_regex=None
[INFO] [tg-signer] 2024-10-25 12:29:37,035 core.py 442 发送文本：V me 50
[INFO] [tg-signer] 2024-10-25 12:30:02,726 core.py 439 匹配到监控项：MatchConfig(chat_id=-4573702599, rule=regex, rule_value=参与关键词：「.*?」), default_send_text=None, send_text_search_regex=参与关键词：「(?P<keyword>(.*?))」\n
[INFO] [tg-signer] 2024-10-25 12:30:02,727 core.py 442 发送文本：我要抽奖
[INFO] [tg-signer] 2024-10-25 12:30:03,001 core.py 226 Message「我要抽奖」 to -4573702599 will be deleted after 5 seconds.
[INFO] [tg-signer] 2024-10-25 12:30:03,001 core.py 229 Waiting...
[INFO] [tg-signer] 2024-10-25 12:30:08,260 core.py 232 Message「我要抽奖」 to -4573702599 deleted!
```

### Version Change Log

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
    *   UDP
    *   HTTP
*   将kurirogram替换为kurigram

#### 0.7.0

*   支持每个聊天会话按序执行多个动作，动作类型：
    *   发送文本
    *   发送骰子
    *   按文本点击键盘
    *   通过图片选择选项
    *   通过计算题回复

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

Configuration and data are stored in the `.signer` directory. You can view the directory structure with:

```bash
tree .signer
```

You will see:

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