# tg-signer: Automate Telegram Tasks with Python

**Automate your Telegram interactions with tg-signer, a versatile Python tool for automated sign-ins, message monitoring, and intelligent responses.  [Check out the original repository](https://github.com/amchii/tg-signer)!**

## Key Features:

*   **Automated Sign-ins:**  Configure daily or scheduled sign-ins with random delays.
*   **Keyboard Interaction:**  Automate clicking buttons based on configured text or AI-powered image recognition.
*   **Message Monitoring and Response:**  Monitor personal chats, groups, and channels, with options for forwarding and automated replies.
*   **Action Flows:**  Define complex action sequences for sign-ins and responses.
*   **AI Integration:**  Leverage AI for image recognition and solving calculation problems within Telegram.
*   **Multi-Account Support:**  Run tasks with multiple Telegram accounts simultaneously.
*   **Flexible Scheduling:**  Use crontab expressions for precise task scheduling.
*   **Message Forwarding:**  Forward messages to external services via UDP and HTTP.

## Installation

Requires Python 3.9 or higher.

**Install using pip:**

```bash
pip install -U tg-signer
```

**For faster performance:**

```bash
pip install "tg-signer[speedup]"
```

**Using Docker:**

Build your own Docker image using the `Dockerfile` and instructions in the [docker](./docker) directory.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <SUBCOMMAND> --help for usage instructions

Subcommand Aliases:
  run_once -> run-once
  send_text -> send-text

Options:
  -l, --log-level [debug|info|warn|error]
                                  Log level, `debug`, `info`, `warn`, `error`
                                  [default: info]
  --log-file PATH                 Log file path, can be a relative path
                                  [default: tg-signer.log]
  -p, --proxy TEXT                Proxy address, e.g.: socks5://127.0.0.1:1080,
                                  will overwrite the `TG_PROXY` environment
                                  variable  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path  [default: .]
  -a, --account TEXT              Custom account name, the session file name
                                  corresponds to <account>.session  [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and sign-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String, will overwrite the
                                  `TG_SESSION_STRING` environment variable
                                  [env var: TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory,
                                  defaults to False, stored in file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to terminal.
  import                  Import configuration, defaults to read from the terminal.
  list                    List existing configurations
  list-members            Query the members of a chat (group or channel),
                          channel requires administrator privileges
  list-schedule-messages  Display configured scheduled messages
  login                   Login to the account (used to get the session)
  logout                  Log out of the account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts simultaneously using one set of
                          configurations
  reconfig                Reconfigure
  run                     Run sign-in according to the task configuration
  run-once                Run a sign-in task once, even if the sign-in task has
                          been executed today
  schedule-messages       Batch configure Telegram's built-in scheduled message
                          function
  send-text               Send a message once, make sure that the current
                          session has "seen" the `chat_id`
  version                 Show version
```

### Examples:

```bash
tg-signer run
tg-signer run my_sign  # Run 'my_sign' task directly without prompting
tg-signer run-once my_sign  # Run 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id '8671234001'
tg-signer send-text -- -10006758812 浇水  # Use '--' for negative chat IDs
tg-signer send-text --delete-after 1 8671234001 /test  # Send and delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages
tg-signer monitor run  # Configure message monitoring and auto-reply
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' for multiple accounts
```

### Configuring a Proxy (if needed)

`tg-signer` does not read system proxy settings. Configure your proxy using the `TG_PROXY` environment variable or the `--proxy` command-line argument.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. The login process retrieves recent chats, ensuring your desired chat is available.

### Sending a Single Message

```bash
tg-signer send-text 8671234001 hello  # Sends 'hello' to chat_id '8671234001'
```

### Running a Sign-in Task

```bash
tg-signer run
```

Or run a specific task:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure the task.

#### Example Sign-in Configuration:

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

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure your monitor.

#### Example Monitoring Configuration:

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

#### Monitoring Example Explanation:

1.  Both chat IDs and user IDs support integer IDs and usernames (usernames must start with "@").

2.  Matching rules (case-insensitive):

    *   `exact`: Exact match required.
    *   `contains`: Message contains the rule value (e.g., "kfc").
    *   `regex`: Regular expression matching.
    *   `all`: Matches all messages.

3.  Additional monitoring features:

    *   Filter by specific user IDs.
    *   Set default text to send upon a match.
    *   Extract text using regex.

4. Message Structure example.

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

#### Example Monitoring Output:

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

### Version Changelog

#### 0.7.6

*   fix: Monitoring messages to multiple chats (#55)

#### 0.7.5

*   Catch and record all RPC errors during task execution
*   bump kurigram version to 2.2.7

#### 0.7.4

*   Support fixed time interval when executing multiple actions
*   No longer limits the number of times to execute per day when configuring regular execution via `crontab`

#### 0.7.2

*   Support forwarding messages to external endpoints, through:
    *   UDP
    *   HTTP
*   Replace kurirogram with kurigram

#### 0.7.0

*   Support for sequentially executing multiple actions per chat session, action types:
    *   Send text
    *   Send dice
    *   Click on keyboard by text
    *   Select options by image
    *   Reply by calculation questions

#### 0.6.6

*   Added support for sending DICE messages

#### 0.6.5

*   Fixed the issue of sign-in records being shared when running multiple accounts with the same configuration

#### 0.6.4

*   Added support for simple calculation questions
*   Improved sign-in configuration and message processing

#### 0.6.3

*   Compatible with breaking changes in kurigram 2.1.38
    > Remove coroutine param from run method [a7afa32](https://github.com/KurimuzonAkuma/pyrogram/commit/a7afa32df208333eecdf298b2696a2da507bde95)

#### 0.6.2

*   Ignore chats that fail to send messages during sign-in

#### 0.6.1

*   Support image recognition after clicking the button text

#### 0.6.0

*   Signer supports timing via crontab
*   Monitor adds `all` support for all messages
*   Monitor supports pushing through server sauce after matching messages
*   Signer adds `multi-run` to run multiple accounts with one set of configurations

#### 0.5.2

*   Monitor supports configuring AI to reply to messages
*   Added the function of batch configuration of "Telegram's built-in scheduled message function"

#### 0.5.1

*   Add `import` and `export` commands for importing and exporting configurations

#### 0.5.0

*   Click the keyboard by the configured text
*   Call AI to identify images and click the keyboard

### Configuration and Data Storage

Data and configurations are stored in the `.signer` directory by default.  Use `tree .signer` to view the structure:

```
.signer
├── latest_chats.json  # Recently obtained conversations
├── me.json  # Personal information
├── monitors  # Monitoring
│   ├── my_monitor  # Monitoring task name
│       └── config.json  # Monitoring configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in record

3 directories, 4 files