# TG-Signer: Automate Telegram Tasks with Ease

Tired of manually interacting with Telegram?  TG-Signer is your all-in-one Python tool for automated Telegram actions, including daily check-ins, message monitoring, and intelligent auto-replies.  [View the original repo on GitHub](https://github.com/amchii/tg-signer).

## Key Features

*   **Automated Check-ins:** Schedule and perform daily sign-in tasks with customizable timing.
*   **Intelligent Actions:**  Automate actions like clicking keyboard buttons based on text or image recognition using AI.
*   **Advanced Monitoring:**  Monitor personal chats, groups, and channels, and trigger actions based on message content, including forwarding to external endpoints.
*   **Flexible Configuration:**  Define complex action flows with multiple steps and conditional logic.
*   **Multi-Account Support:**  Run tasks with multiple Telegram accounts simultaneously.
*   **AI-Powered Automation:**  Leverage AI for image recognition and advanced message replies.
*   **Message Scheduling:** Leverage the Telegram built-in message scheduling features.
*   **Extensible with External Services**: Forward messages via UDP/HTTP.

## Installation

**Prerequisites:** Python 3.9 or higher

Install with pip:

```bash
pip install -U tg-signer
```

For faster performance, install with the speedup option:

```bash
pip install "tg-signer[speedup]"
```

### Docker (Optional)

Build a Docker image using the `Dockerfile` and related documentation in the [`docker`](./docker) directory of the original repo.

## Usage

### Commands

Use `<subcommand> --help` to view detailed usage instructions for each command.  Alias commands include: `run_once` as `run-once` and `send_text` as `send-text`.

Key commands include:

*   `run`: Run a configured sign-in task.
*   `run-once`: Run a sign-in task once, regardless of previous executions.
*   `send-text`: Send a text message to a chat.
*   `monitor`: Configure and run message monitoring and auto-reply tasks.
*   `multi-run`: Run the same task with multiple accounts.
*   `login`: Login to Telegram.

### Examples

*   `tg-signer run`: Run a configured sign-in task.
*   `tg-signer run my_sign`: Run the task "my_sign".
*   `tg-signer send-text 8671234001 /test`: Send "/test" to chat ID 8671234001.
*   `tg-signer monitor run`: Configure and start a monitoring task.
*   `tg-signer multi-run -a account_a -a account_b same_task`: Run 'same_task' with accounts 'account_a' and 'account_b'.

### Configuration

*   **Proxy:** Configure a proxy using the `TG_PROXY` environment variable or the `--proxy` command-line option (e.g., `export TG_PROXY=socks5://127.0.0.1:7890`).
*   **Login:** Use `tg-signer login` to log in and retrieve your chat list.

## Configuration and Data Storage

Configuration files and data are stored in the `.signer` directory within your working directory.

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

## Version Changelog

A detailed changelog is available in the original README.

---