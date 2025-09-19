# 📚 Calibre-Web Automated Book Downloader: Effortlessly Download and Manage Your eBooks

**Streamline your eBook workflow!** Calibre-Web Automated Book Downloader provides an intuitive web interface for searching, requesting, and downloading books, seamlessly integrating with [Calibre-Web-Automated](https://github.com/calibrain/calibre-web-automated).

## ✨ Key Features

*   🌐 **User-Friendly Interface:** Easily search and request book downloads via a simple web interface.
*   🔄 **Automated Downloads:** Downloads books directly to your specified ingest folder, ready for Calibre.
*   🔌 **Seamless Integration:** Designed to work perfectly with Calibre-Web-Automated for a complete eBook management solution.
*   📖 **Multi-Format Support:** Supports a wide range of eBook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:**  Includes a built-in Cloudflare bypass, or use your own external resolver, for more reliable downloads.
*   🐳 **Dockerized Deployment:** Quick and easy setup with Docker and Docker Compose.
*   🧅 **Tor Variant:**  Offers a Tor-enabled variant for enhanced privacy and bypassing network restrictions.
*   ☁️ **External Cloudflare Resolver:** Integrate with external Cloudflare resolvers such as FlareSolverr.

## 🚀 Getting Started

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation

1.  **Get the `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:** Navigate to `http://localhost:8084` in your web browser.

## ⚙️ Configuration

### Environment Variables

Configure the application's behavior using environment variables. Key settings are outlined below:

#### Application Settings

| Variable          | Description             | Default Value      |
| ----------------- | ----------------------- | ------------------ |
| `FLASK_PORT`      | Web interface port      | `8084`             |
| `FLASK_HOST`      | Web interface binding   | `0.0.0.0`          |
| `DEBUG`           | Debug mode toggle       | `false`            |
| `INGEST_DIR`      | Book download directory | `/cwa-book-ingest` |
| `TZ`              | Container timezone      | `UTC`              |
| `UID`             | Runtime user ID         | `1000`             |
| `GID`             | Runtime group ID        | `100`              |
| `CWA_DB_PATH`     | Calibre-Web's database  | None               |
| `ENABLE_LOGGING`  | Enable log file         | `true`             |
| `LOG_LEVEL`       | Log level to use        | `info`             |

**Authentication:** To enable authentication, set `CWA_DB_PATH` to your Calibre-Web database file (`app.db`).

**Logging:** Logs are saved to `/var/log/cwa-book-downloader` (if enabled).

#### Download Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum retry attempts                                    | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                                     | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                           | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                                    | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books                              | `en`                              |
| `AA_DONATOR_KEY`       | Optional Donator key for Anna's Archive fast download API | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID                  | `false`                           |
| `PRIORITIZE_WELIB`     | When downloading, download from WELIB first instead of AA | `false`                           |

**Multiple Languages:**  Set `BOOK_LANGUAGE` to a comma-separated list (e.g., `en,fr,ru`).

#### Anna's Archive Settings (AA)

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

**Anna's Archive Donator Key:**  Use your key in `AA_DONATOR_KEY` for faster downloads.
**Cloudflare Bypass:** Disable the Cloudflare bypass by setting `USE_CF_BYPASS` to false to rely on alternative download hosts.

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

**Proxy Configuration:** Configure HTTP/HTTPS proxies using the following format:
```bash
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080

# With Authentication
HTTP_PROXY=http://username:password@proxy.example.com:8080
HTTPS_PROXY=http://username:password@proxy.example.com:8080
```

**Custom DNS:** Use custom DNS servers, or use pre-set providers like Google, Cloudflare, or OpenDNS.

```bash
CUSTOM_DNS=cloudflare
USE_DOH=true # For DNS over HTTPS with supported providers.
```

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

**Post-Download Script:** Set `CUSTOM_SCRIPT` to execute a custom script after each download.  The script receives the downloaded file path as an argument.

```yaml
environment:
  - CUSTOM_SCRIPT=/scripts/process-book.sh
volumes:
  - local/scripts/custom_script.sh:/scripts/process-book.sh
```

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Note:** For CIFS shares, add `nobrl` to your fstab mount options to avoid "database locked" errors.

## Variants

### 🧅 Tor Variant

For enhanced privacy, the Tor variant routes all traffic through the Tor network.

1.  **Get the Tor Compose File:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  **Start the Tor Service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

**Important Tor Considerations:**  This variant requires `NET_ADMIN` and `NET_RAW` Docker capabilities.  Timezone is set by the Tor exit node IP.  Custom DNS and proxy settings are ignored.

### ☁️ External Cloudflare Resolver Variant

Use an external service to bypass Cloudflare protection.

1.  **Get the External Resolver Compose File:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  **Start the External Resolver Service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

**Compatibility:**  Works with resolvers that implement the FlareSolverr API schema.

## 🏗️ Architecture

The application consists of a single service:

1.  **calibre-web-automated-bookdownloader:** Main application providing web interface and download functionality

## 🏥 Health Monitoring

Built-in health checks monitor the web interface, download service, and Cloudflare bypass service.

## 📝 Logging

Logs are available in:

*   Container: `/var/logs/cwa-book-downloader.log`
*   Docker logs: Access via `docker logs`

## 🤝 Contributing

Contributions are welcome!  Please submit Pull Requests.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

This tool is designed for legitimate use only. Users are responsible for ensuring they have the right to download copyrighted materials and for complying with copyright laws and regulations.

### Duplicate Downloads Warning

The current version **does not** check for existing files or books in your Calibre database. Be cautious when requesting multiple books to avoid duplicates.

## 💬 Support

For questions or to report issues, please file an issue on the GitHub repository.
```

Key improvements and SEO considerations:

*   **Clear, concise title:**  "Calibre-Web Automated Book Downloader" is included.
*   **One-sentence hook:**  Emphasizes the primary benefit.
*   **Keyword-rich headings:**  Includes "eBook", "Download", "Calibre-Web", and other relevant terms in headings and subheadings.
*   **Bulleted key features:**  Highlights the value proposition.
*   **Concise explanations:**  The installation instructions and configuration details are streamlined.
*   **Emphasis on benefits:**  The text constantly reminds the reader *why* they should use the tool.
*   **Action-oriented language:**  Uses phrases like "Effortlessly Download", "Streamline", and "Get Started".
*   **Complete, self-contained:**  The README is a complete and easily understood document.
*   **Clear structure:**  Uses Markdown effectively for readability.
*   **Relevant keywords:** Keywords like "eBook", "download", "Calibre", "automated", "Docker", "book management", "Anna's Archive" are included.
*   **Variants section:** Highlights advanced capabilities and options.
*   **Emphasis on usage:** Instructions and configuration details are provided to showcase the ease of use.
*   **Important Disclaimers:** Added key disclaimers on copyright and duplicate downloads to reduce potential issues.
*   **Calls to action:** Encourages the reader to "Get Started" and contribute.
*   **Health checks & architecture sections:** Included in the README to provide technical insights.
*   **Tor variant description:** Expanded Tor variant descriptions to include prerequisites and considerations.