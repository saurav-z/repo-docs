# tg-signer: Automate Your Telegram Tasks

**Effortlessly automate Telegram activities like daily check-ins, message monitoring, and auto-replies with tg-signer!**  [View the GitHub Repository](https://github.com/amchii/tg-signer)

## Key Features

*   **Automated Check-ins:** Schedule and automate daily check-ins with customizable timing.
*   **Intelligent Interactions:** Interact with Telegram chats using AI-powered image recognition and keyboard clicks based on text or AI analysis.
*   **Message Monitoring and Auto-Reply:** Monitor personal chats, groups, and channels, with options for forwarding and automatic replies.
*   **Action Flows:** Define and execute complex action flows for diverse automation scenarios.
*   **Flexible Configuration:** Configure tasks via command line or with environment variables.
*   **Multi-Account Support:** Run multiple accounts concurrently using the same configuration.
*   **AI-Powered Responses:** Leverage AI for image analysis, calculation problems, and message replies.

## Installation

**Prerequisites:** Python 3.9 or higher

Install using pip:

```bash
pip install -U tg-signer
```

For faster performance, install with optional dependencies:

```bash
pip install "tg-signer[speedup]"
```

### Docker

For Docker users, build your own image using the `Dockerfile` and follow the instructions in the [docker/README.md](./docker/README.md) file.

## Usage

```bash
tg-signer [OPTIONS] COMMAND [ARGS]...
```

Use `<subcommand> --help` to view specific command usage.

**Example Commands:**

*   `tg-signer run`: Run a configured sign-in task.
*   `tg-signer run my_sign`: Run a specific sign-in task named "my\_sign" directly.
*   `tg-signer send-text 8671234001 /test`: Send the text "/test" to the chat with the ID 8671234001.
*   `tg-signer monitor run`: Configure and run message monitoring and auto-reply.
*   `tg-signer multi-run -a account_a -a account_b same_task`: Run `same_task` for both `account_a` and `account_b`.

## Configuration

*   **Proxy:** Configure proxy settings using the `TG_PROXY` environment variable or the `--proxy` command-line option.

    ```bash
    export TG_PROXY=socks5://127.0.0.1:7890
    ```

*   **Login:**

    ```bash
    tg-signer login
    ```

    Follow the prompts to enter your phone number and verification code to log in.

## Core Functionality

### Running a Sign-in Task

```bash
tg-signer run
```

Or, specify a task name directly:

```bash
tg-signer run linuxdo
```

The system will then prompt you to configure the task.

**Example Sign-in Configuration:**

```
... (Example configuration steps will be detailed in the actual usage, see original README) ...
```

### Configuring and Running Monitoring

```bash
tg-signer monitor run my_monitor
```

The system will then prompt you to configure the monitoring task.

**Example Monitoring Configuration:**

```
... (Example configuration steps will be detailed in the actual usage, see original README) ...
```

## Configuration and Data Storage

Configuration and data are stored in the `.signer` directory. The structure looks like this:

```
.signer
├── latest_chats.json  # Latest conversations
├── me.json  # Personal information
├── monitors  # Monitoring tasks
│   ├── my_monitor  # Monitoring task name
│   │   └── config.json  # Monitoring configuration
└── signs  # Sign-in tasks
    └── linuxdo  # Sign-in task name
        ├── config.json  # Sign-in configuration
        └── sign_record.json  # Sign-in record

3 directories, 4 files
```

## Version Changelog

**(See Original README for specific version updates.)**
```
... (Version Change Log from original README) ...