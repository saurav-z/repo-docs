# tg-signer: Automate Your Telegram Tasks

**Automate Telegram tasks like daily check-ins, message monitoring, and auto-replies with tg-signer, a versatile Python-based tool.**  [View the GitHub Repository](https://github.com/amchii/tg-signer)

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with flexible time offsets.
*   **Interactive Automation:** Click buttons based on configured text or AI-powered image recognition.
*   **Intelligent Message Handling:** Monitor, forward, and auto-reply to messages in individual chats, groups, and channels.
*   **Configurable Action Flows:** Execute complex workflows based on your specific needs.
*   **Flexible Deployment:** Install via pip or use Docker for easy setup.
*   **Multi-Account Support:** Run multiple Telegram accounts concurrently.
*   **Message Scheduling:** Schedule messages using Telegram's built-in scheduling feature.
*   **AI Integration:** Leverage AI for image recognition and solving calculation problems.
*   **Message Forwarding:** Forward messages via UDP or HTTP.

## Installation

**Prerequisites:** Python 3.9 or higher

**Install using pip:**

```bash
pip install -U tg-signer
```

**For improved performance:**

```bash
pip install "tg-signer[speedup]"
```

**Docker:**
Build your own image using the `Dockerfile` and accompanying README located in the [docker](./docker) directory.

## Getting Started

### Basic Usage

```bash
tg-signer run
tg-signer run my_sign  # Run 'my_sign' task directly, without prompts
tg-signer run-once my_sign  # Run 'my_sign' task once, regardless of previous runs
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id 8671234001
tg-signer send-text -- -10006758812 浇水  # Use '--' for negative chat IDs
tg-signer send-text --delete-after 1 8671234001 /test  # Send '/test' and delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 你好  # Schedule messages for 10 days
tg-signer monitor run  # Configure message monitoring and auto-replies
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' with multiple accounts
```

### Configuration

*   **Proxy Configuration:** Configure proxies using the `TG_PROXY` environment variable or the `--proxy` command-line option.

    ```bash
    export TG_PROXY=socks5://127.0.0.1:7890
    ```

*   **Login:**

    ```bash
    tg-signer login
    ```

    Follow the prompts to log in and authorize your account.
### Core Commands

The tool provides a range of commands to handle various functionalities. The below command is only illustrative, for detailed explanation, please see the help documents.

| Command | Description  |
| :---------------- |:----------------  |
| `run` | Run the scheduled or configured sign-in tasks.  |
| `run-once` | Executes a sign-in task once, ignoring its execution status for the current day.  |
| `send-text` | Sends a text message to the specified chat or user.  |
| `monitor` | Configures and runs the message monitoring and auto-reply system.  |
| `multi-run` | Enables running the same task with multiple accounts simultaneously.  |
| `list` |  Lists available configuration. |
| `import` | Import configuration from the terminal or specified source.  |
| `export` | Exports configuration to the terminal or a specified file.  |
| `login` | Logs in to your Telegram account to get the session.  |
| `logout` | Logs out of your account and removes the session file.  |
| `list-members` | Lists members of a group or channel, requires admin privileges for channels.  |
| `list-schedule-messages` | Displays the currently configured scheduled messages.  |
| `schedule-messages` | Configures the Telegram's built-in message scheduling feature.  |
| `reconfig` | Reconfigures the application settings. |
| `version` |  Shows the current version of the application. |


### Sign-in task
```bash
tg-signer run
```

or specify task name directly
```bash
tg-signer run linuxdo
```
Follow the prompts to configure your sign-in tasks, including specifying the actions to perform, such as sending text, clicking buttons, or utilizing image recognition.

**Example configuration flow**
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

### Message Monitoring & Auto-Reply

```bash
tg-signer monitor run my_monitor
```

Configure rules to monitor messages and trigger automated replies.  The example below illustrates how to set up message monitoring with different rules, including exact, contains, and regex matching, to trigger automated responses and forward messages based on specified criteria.

**Example Configuration**

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

**Important Notes:**
*   **Chat ID and User ID:** Use both integer IDs and usernames, with usernames starting with `@`.
*   **Matching Rules:** `exact`, `contains`, `regex`, and `all` are available, case-insensitive.
*   **Message Structure:** The structure of message data is defined as the example shows, which includes the user and chat details.
*   **Extract Text:** Leverage regex to extract specific information from messages.
*   **AI Reply & Forwarding:** Configure AI-powered responses and message forwarding.

**Example output**
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

### Storage Location

Configuration and data are stored in the `.signer` directory, with the following structure:

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

### Version History

*   **0.7.6:** Fix for message forwarding in multiple chats.
*   **0.7.5:** Improved error handling and kurigram version update.
*   **0.7.4:** Support for fixed time intervals between actions.
*   **0.7.2:** Message forwarding via UDP and HTTP.
*   **0.7.0:** Introduced sequential action flows, including send text, dice, keyboard clicks, image selection, and calculation responses.
*   **0.6.6:** Added support for sending DICE messages.
*   **0.6.5:** Fixed issues with shared sign-in records when running multiple accounts.
*   **0.6.4:** Enabled support for simple calculation problems and enhanced message handling.
*   **0.6.3:** Compatibility update for kurigram version changes.
*   **0.6.2:** Excluded sign-in failures from chat notifications.
*   **0.6.1:** Added support for image recognition after button clicks.
*   **0.6.0:** Sign-in tasks can now be scheduled via crontab. Support for message pushing via Server酱, and 'multi-run' feature
*   **0.5.2:** Enhanced AI response configurations and the ability to use Telegram's built-in scheduled messages.
*   **0.5.1:** Added `import` and `export` commands.
*   **0.5.0:** Added functionality to click buttons based on configured text, use AI to click buttons based on image recognition.