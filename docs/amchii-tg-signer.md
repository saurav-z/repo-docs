# tg-signer: Automate Telegram Tasks with Python

Automate your Telegram tasks with tg-signer, a versatile Python tool for automated sign-ins, message monitoring, and AI-powered interactions.  [View the GitHub Repository](https://github.com/amchii/tg-signer)

## Key Features

*   **Automated Sign-Ins:**
    *   Daily scheduled sign-ins with customizable time offsets.
    *   Click buttons based on configured text.
    *   Use AI for image recognition and button clicks.
*   **Message Monitoring & Automation:**
    *   Monitor personal chats, groups, and channels.
    *   Forward and auto-reply to messages based on rules.
*   **AI-Powered Interactions:**
    *   Utilize AI for image recognition and solving math problems.
*   **Flexible Configuration:**
    *   Configure actions flow for tasks.
    *   Supports custom actions to trigger tasks.
*   **Message Scheduling:**
    *   Schedule messages using cron-like expressions.
*   **Multi-Account Support:**
    *   Run multiple accounts simultaneously.
*   **Versatile Monitoring:**
    *   Monitor chat ID and user ID using integers or usernames.
    *   Support for *exact*, *contains*, *regex* and *all* matching rules.
    *   Reply with a default text, extract text with regex, push using ServerChan, forward to UDP/HTTP endpoints.
*   **Easy Installation & Use:**
    *   Simple installation via pip.
    *   Docker support.
*   **Configuration Export/Import:**
    *   Export and import configurations.

## Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

For faster performance:

```bash
pip install "tg-signer[speedup]"
```

### Docker

Build your own Docker image using the provided `Dockerfile` in the `docker` directory and the Docker README.

## Usage

```
Usage: tg-signer [OPTIONS] COMMAND [ARGS]...

  Use <subcommand> --help to view usage instructions

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
                                  will override the environment variable
                                  `TG_PROXY`  [env var: TG_PROXY]
  --session_dir PATH              Directory to store TG Sessions, can be a
                                  relative path  [default: .]
  -a, --account TEXT              Custom account name, corresponding session
                                  file name is <account>.session  [env var:
                                  TG_ACCOUNT; default: my_account]
  -w, --workdir PATH              tg-signer working directory, used to store
                                  configuration and sign-in records, etc.
                                  [default: .signer]
  --session-string TEXT           Telegram Session String, will override the
                                  environment variable `TG_SESSION_STRING`  [env
                                  var: TG_SESSION_STRING]
  --in-memory                     Whether to store the session in memory, the
                                  default is False, stored in a file
  --help                          Show this message and exit.

Commands:
  export                  Export configuration, defaults to output to the
                          terminal.
  import                  Import configuration, defaults to read from the
                          terminal.
  list                    List existing configurations
  list-members            Query members of a chat (group or channel),
                          channels require administrator privileges
  list-schedule-messages  Display configured scheduled messages
  login                   Login to an account (used to obtain a session)
  logout                  Log out of an account and delete the session file
  monitor                 Configure and run monitoring
  multi-run               Run multiple accounts at the same time using a set of
                          configurations
  reconfig                Reconfigure
  run                     Run sign-in based on task configuration
  run-once                Run a sign-in task once, even if the sign-in task
                          has been executed today
  schedule-messages       Batch configure Telegram's built-in timed message
                          sending function
  send-text               Send a message once, make sure the current session
                          has "seen" the `chat_id`
  version                 Show version
```

**Examples:**

```bash
tg-signer run
tg-signer run my_sign  # Run the 'my_sign' task directly, without prompting
tg-signer run-once my_sign  # Run the 'my_sign' task once
tg-signer send-text 8671234001 /test  # Send '/test' to chat_id 8671234001
tg-signer send-text -- -10006758812 water  # Use '--' before negative chat IDs
tg-signer send-text --delete-after 1 8671234001 /test  # Send and delete after 1 second
tg-signer list-members --chat_id -1001680975844 --admin  # List channel admins
tg-signer schedule-messages --crontab '0 0 * * *' --next-times 10 -- -1001680975844 Hello  # Schedule messages
tg-signer monitor run  # Configure message monitoring and auto-reply
tg-signer multi-run -a account_a -a account_b same_task  # Run 'same_task' config with multiple accounts
```

## Configuration

### Proxy Configuration

`tg-signer` does not read system proxies. Configure using the environment variable `TG_PROXY` or the `--proxy` command-line argument:

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Logging In

```bash
tg-signer login
```

Follow the prompts to enter your phone number and verification code. This process also retrieves your chat list.

### Sending a Single Message

```bash
tg-signer send-text 8671234001 Hello  # Send 'hello' to chat_id 8671234001
```

### Running a Sign-In Task

```bash
tg-signer run
```

Or, specify the task name:

```bash
tg-signer run linuxdo
```

Configure the task as prompted.

**Example Configuration:**

```
Begin to configure the task <linuxdo>
1. Chat ID (ID from login recent dialog output): 7661096533
2. Chat Name (optional): jerry bot
3. Start configuring <action>, configure according to the actual sign-in order.
    1: Send normal text
    2: Send Dice emoji
    3: Click keyboard based on text
    4: Select option based on picture
    5: Reply to a calculation question

1st action:
1. Enter the corresponding number to select the action: 1
2. Enter the text to be sent: checkin
3. Continue adding actions? (y/N): y
2nd action:
1. Enter the corresponding number to select the action: 3
2. Text to be clicked in the keyboard: sign in
3. Continue adding actions? (y/N): y
3rd action:
1. Enter the corresponding number to select the action: 4
Image recognition will use a large model to answer, please make sure the large model supports image recognition.
2. Continue adding actions? (y/N): y
4th action:
1. Enter the corresponding number to select the action: 5
Calculation questions will use the large model to answer.
2. Continue adding actions? (y/N): y
5th action:
1. Enter the corresponding number to select the action: 2
2. Enter dice to send (e.g. 🎲, 🎯): 🎲
3. Continue adding actions? (y/N): n
Please set `OPENAI_API_KEY`, `OPENAI_BASE_URL` through environment variables before running. The default model is "gpt-4o", which can be changed by environment variable `OPENAI_MODEL`.
4. Wait N seconds after sending the message to delete (wait to delete after sending the message, '0' means delete immediately, no need to delete directly return), N: 10
╔════════════════════════════════════════════════╗
║ Chat ID: 7661096533                            ║
║ Name: jerry bot                                ║
║ Delete After: 10                               ║
╟────────────────────────────────────────────────╢
║ Actions Flow:                                  ║
║ 1. [Send normal text] Text: checkin            ║
║ 2. [Click keyboard based on text] Click: sign in  ║
║ 3. [Select option based on picture]             ║
║ 4. [Reply to a calculation question]            ║
║ 5. [Send Dice emoji] Dice: 🎲                    ║
╚════════════════════════════════════════════════╝
The 1st sign-in is configured successfully

Continue configuring sign-in? (y/N): n
Daily sign-in time (time or crontab expression, such as '06:00:00' or '0 6 * * *'):
Sign-in time error random seconds (default is 0): 300
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

Configure as prompted.

**Example Configuration:**

```
Begin to configure the task <my_monitor>
Both chat id and user id support integer id and string username, username must start with @, such as @neo

Configure the 1st monitoring item
1. Chat ID (ID from login recent dialog output): -4573702599
2. Matching rule ('exact', 'contains', 'regex', 'all'): contains
3. Rule value (cannot be empty): kfc
4. Only match messages from specific user ID (separate multiple with commas, just return to match all users): @neo
5. Default send text: V Me 50
6. Extract send text regular expression from message:
7. Wait N seconds after sending the message to delete (wait to delete after sending the message, '0' means delete immediately, no need to delete directly return), N:
Continue configuring? (y/N): y

Configure the 2nd monitoring item
1. Chat ID (ID from login recent dialog output): -4573702599
2. Matching rule ('exact', 'contains', 'regex'): regex
3. Rule value (cannot be empty): 参与关键词：「.*?」
4. Only match messages from specific user ID (separate multiple with commas, just return to match all users): 61244351
5. Default send text:
6. Extract send text regular expression from message: 参与关键词：「(?P<keyword>(.*?))」\n
7. Wait N seconds after sending the message to delete (wait to delete after sending the message, '0' means delete immediately, no need to delete directly return), N: 5
Continue configuring? (y/N): y

Configure the 3rd monitoring item
1. Chat ID (ID from login recent dialog output): -4573702599
2. Matching rule(exact, contains, regex, all): all
3. Only match messages from specific user ID (separate multiple with commas, just return to match all users):
4. Always ignore messages sent by yourself (y/N): y
5. Default send text (enter to skip):
6. Use AI for reply(y/N): n
7. Extract send text regular expression from message (enter directly if not needed):
8. Push messages via Server Chan(y/N): n
9. Need to forward to external(UDP, Http)(y/N): y
10. Need to forward to UDP(y/N): y
11. Please enter UDP server address and port (e.g. `127.0.0.1:1234`): 127.0.0.1:9999
12. Need to forward to Http(y/N): y
13. Please enter Http address (e.g. `http://127.0.0.1:1234`): http://127.0.0.1:8000/tg/user1/messages
Continue configuring? (y/N): n
```

**Example Explanation:**

1.  Both `chat id` and `user id` support **integer IDs** and **usernames**.  Usernames *must begin with "@"*. Example: "@neo".  Note that the *username* may not exist.
2.  Matching Rules (case-insensitive):
    *   `exact`: Exact match of the message.
    *   `contains`: Message contains the specified value (e.g., "kfc").
    *   `regex`:  Regular expression match.  See [Python Regular Expressions](https://docs.python.org/zh-cn/3/library/re.html).
    *   `all`: Matches all incoming messages.
3.  Message Structure (example):

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

**Example Output:**

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

## Version Changelog

A complete changelog is available in the original README.

## Configuration & Data Storage

Configuration and data are stored in the `.signer` directory.  The structure is as follows:

```
.signer
├── latest_chats.json  # Recently obtained chats
├── me.json  # Personal information
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitoring task name
│       └── config.json  # Monitoring configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in record

3 directories, 4 files
```