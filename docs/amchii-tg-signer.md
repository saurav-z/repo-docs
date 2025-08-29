# tg-signer: Automate Your Telegram Tasks with Python

**Effortlessly automate Telegram interactions for tasks like daily check-ins, message monitoring, and auto-replies with this versatile Python tool.** Learn more and contribute at the [original repo](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Check-ins:** Schedule daily check-ins with random time offsets.
*   **Interactive Actions:** Configure actions to send texts, interact with keyboards via text, and use AI for image recognition and keyboard interactions.
*   **Message Monitoring and Auto-Reply:** Monitor personal chats, groups, and channels, then forward or automatically reply based on custom rules.
*   **Customizable Action Flows:** Define action flows for complex interactions.
*   **Flexible Configuration:** Uses environment variables and command-line arguments for easy configuration.
*   **Multi-Account Support:** Manage multiple Telegram accounts.
*   **Message Scheduling:** Utilize Telegram's built-in message scheduling.

## Installation

Requires Python 3.9 or higher.

```bash
pip install -U tg-signer
```

To improve program speed:

```bash
pip install "tg-signer[speedup]"
```

**Docker:**  You can build your own Docker image using the provided [Dockerfile](./docker/Dockerfile) and [README](./docker/README.md) in the `docker` directory.

## Usage

Use the `<subcommand> --help` command to view usage instructions.

```bash
tg-signer [OPTIONS] COMMAND [ARGS]...
```

**Example Commands:**

*   `tg-signer run`: Runs a configured check-in task.
*   `tg-signer run my_sign`: Runs the 'my\_sign' task immediately.
*   `tg-signer run-once my_sign`: Runs the 'my\_sign' task once, regardless of whether it's already been executed today.
*   `tg-signer send-text 8671234001 /test`: Sends the text "/test" to a chat with the ID "8671234001".
*   `tg-signer monitor run`: Configures and runs message monitoring and auto-reply.
*   `tg-signer multi-run -a account_a -a account_b same_task`: Runs 'same_task' for 'account\_a' and 'account\_b' simultaneously.

**Available Commands:**

*   `export`: Export configuration.
*   `import`: Import configuration.
*   `list`: List existing configurations.
*   `list-members`: Query chat members.
*   `list-schedule-messages`: Display scheduled messages.
*   `login`: Login to your Telegram account.
*   `logout`: Logout and delete the session file.
*   `monitor`: Configure and run monitoring.
*   `multi-run`: Run tasks with multiple accounts.
*   `reconfig`: Reconfigure.
*   `run`: Run check-in tasks based on configuration.
*   `run-once`: Run a check-in task once.
*   `schedule-messages`: Configure Telegram's scheduled messages feature.
*   `send-text`: Send a message.
*   `version`: Show version.

### Configure Proxy

Configure proxy using the `TG_PROXY` environment variable or the `--proxy` command parameter.

```bash
export TG_PROXY=socks5://127.0.0.1:7890
```

### Login

```bash
tg-signer login
```

Follow the prompts to log in with your phone number and verification code. This will retrieve recent chats.

### Send a Message

```bash
tg-signer send-text 8671234001 hello
```

### Run a Check-in Task

```bash
tg-signer run
```

Or specify a task name:

```bash
tg-signer run linuxdo
```

Follow the prompts to configure.

### Configure and Run Monitoring

```bash
tg-signer monitor run my_monitor
```

Follow the prompts to configure.

### Configuration and Data Storage Location

Data and configurations are stored in the `.signer` directory. The structure is as follows:

```
.signer
├── latest_chats.json  # Latest chats
├── me.json  # Personal information
├── monitors  # Monitoring configurations
│   ├── my_monitor
│       └── config.json  # Monitoring configuration
└── signs  # Check-in task configurations
    └── linuxdo
        ├── config.json  # Check-in configuration
        └── sign_record.json  # Check-in records
```

### Version Changelog

(Changelog information is retained as provided)
```
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