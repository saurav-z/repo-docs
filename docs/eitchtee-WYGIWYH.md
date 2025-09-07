<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: What You Get Is What You Have
  <br>
</h1>

<h4 align="center">Take control of your finances with a simple, powerful, and no-budget approach.</h4>

<p align="center">
  <a href="#about">About</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#environment-variables">Environment Variables</a> •
  <a href="#oidc-configuration">OIDC Configuration</a> •
  <a href="#translation">Translation</a> •
  <a href="#caveats">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH (What You Get Is What You Have)** is an open-source finance tracker designed for those who prefer a straightforward, no-budget approach to money management. Simplify your finances with multi-currency support, customizable transactions, and a built-in dollar-cost averaging tracker.

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## #About
WYGIWYH helps you manage your finances without the complexities of traditional budgeting.  Based on the principle of spending what you earn each month, WYGIWYH offers simplicity and flexibility. Easily track multiple currencies, accounts, and investments without the constraints of a rigid budget.  The goal is to provide a user-friendly finance tracking tool that combines simplicity and power.

## Key Features

*   **Unified Transaction Tracking:**  Record all income and expenses in one place.
*   **Multi-Account Support:** Track funds across banks, wallets, and investments.
*   **Multi-Currency Support:** Manage transactions and balances in different currencies.
*   **Custom Currencies:** Create currencies for crypto, rewards points, or other models.
*   **Automated Adjustments:**  Use customizable rules to automatically modify transactions.
*   **Dollar-Cost Average (DCA) Tracker:**  Track recurring investments (crypto, stocks).
*   **API Support:** Integrate with other services to automate transactions.

## Demo

Experience WYGIWYH firsthand with our demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

Use these credentials (data resets daily):

>   Email: `demo@demo.com`
>
>   Password: `wygiwyhdemo`

>   **Important:** Demo data is reset daily, and some features are disabled.

## How to Use

To run WYGIWYH, you'll need [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).

1.  **Create a WYGIWYH Directory (Optional):**
    ```bash
    mkdir WYGIWYH
    cd WYGIWYH
    ```

2.  **Create `docker-compose.yml`:**
    ```bash
    touch docker-compose.yml
    nano docker-compose.yml
    ```

    *   Paste the contents of `https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml` and adjust as needed.

3.  **Create a `.env` file:**
    ```bash
    touch .env
    nano .env
    ```

    *   Paste the contents of `https://github.com/eitchtee/WYGIWYH/blob/main/.env.example` and configure accordingly.

4.  **Run the Application:**
    ```bash
    docker compose up -d
    ```

5.  **Create Admin Account (if environment variables aren't set):**
    ```bash
    docker compose exec -it web python manage.py createsuperuser
    ```

### Running Locally

1.  In your `.env` file:
    *   Remove `URL`
    *   Set `HTTPS_ENABLED` to `false`
    *   Leave default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1])

2.  Access the application at `localhost:OUTBOUND_PORT`

>   **Important Notes:**
>
>   *   Add your machine's IP to `DJANGO_ALLOWED_HOSTS` if using services like Tailscale.
>   *   Add your IP (without `http://`) to `DJANGO_ALLOWED_HOSTS` when using a non-localhost IP.

### Latest Changes

To access the newest features, build from source or use the `:nightly` tag on Docker.

Find the required Dockerfiles [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

### Unraid

WYGIWYH is available in the Unraid Store.  You will need to provide your own PostgreSQL database (version 15+).  See [nwithan8's Unraid template repo](https://github.com/nwithan8/unraid_templates).

To create an admin user via Unraid's UI, click the WYGIWYH icon on the Docker page, select `Console`, and run `python manage.py createsuperuser`.

## Environment Variables

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                                                             |
|-------------------------------|-------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DJANGO_ALLOWED_HOSTS`          | string      | localhost 127.0.0.1               | Space-separated domains and IPs that the WYGIWYH site can serve.  ([Django Docs](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts))                                                                                                             |
| `HTTPS_ENABLED`                 | true\|false | false                             | Enables secure cookies.                                                                                                                                                                                                                                                 |
| `URL`                           | string      | http://localhost http://127.0.0.1 | Space-separated domains and IPs (with protocol) representing trusted origins for unsafe requests (e.g., POST). ([Django Docs](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins))                                                            |
| `SECRET_KEY`                    | string      | ""                                | Cryptographic signing key (unique and unpredictable).                                                                                                                                                                                                                 |
| `DEBUG`                         | true\|false | false                             | Enables or disables debug mode (don't use in production).                                                                                                                                                                                                                |
| `SQL_DATABASE`                  | string      | None *required                    | PostgreSQL database name.                                                                                                                                                                                                                                               |
| `SQL_USER`                      | string      | user                              | PostgreSQL username.                                                                                                                                                                                                                                                    |
| `SQL_PASSWORD`                  | string      | password                          | PostgreSQL password.                                                                                                                                                                                                                                                    |
| `SQL_HOST`                      | string      | localhost                         | PostgreSQL host address.                                                                                                                                                                                                                                                  |
| `SQL_PORT`                      | string      | 5432                              | PostgreSQL port.                                                                                                                                                                                                                                                      |
| `SESSION_EXPIRY_TIME`           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                                                          |
| `ENABLE_SOFT_DELETE`            | true\|false | false                             | Enables soft-deletion of transactions (deleted transactions remain in the database). Useful for imports.                                                                                                                                                              |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | 365                               | Time (in days) to retain soft-deleted transactions (requires `ENABLE_SOFT_DELETE`).  If 0, transactions are kept indefinitely.                                                                                                                                            |
| `TASK_WORKERS`                  | int         | 1                                 | Number of workers for asynchronous tasks.                                                                                                                                                                                                                              |
| `DEMO`                          | true\|false | false                             | Enables or disables demo mode.                                                                                                                                                                                                                                         |
| `ADMIN_EMAIL`                   | string      | None                              | Automatically creates an admin account with this email (requires `ADMIN_PASSWORD`).                                                                                                                                                                                   |
| `ADMIN_PASSWORD`                | string      | None                              | Automatically creates an admin account with this password (requires `ADMIN_EMAIL`).                                                                                                                                                                                  |
| `CHECK_FOR_UPDATES`             | bool        | true                              | Checks for and notifies users of new versions (queries GitHub API every 12 hours).                                                                                                                                                                              |

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) for login via `django-allauth`.

>   **Note:** Only OpenID Connect is supported.

Configure these environment variables:

| Variable             | Description                                                                                                                                                                                                      |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | Provider name (shown on the login page).  Defaults to `OpenID Connect`.                                                                                                                                      |
| `OIDC_CLIENT_ID`     | OIDC provider's Client ID.                                                                                                                                                                                      |
| `OIDC_CLIENT_SECRET` | OIDC provider's Client Secret.                                                                                                                                                                                  |
| `OIDC_SERVER_URL`    | OIDC provider's discovery document or authorization server base URL (e.g., `https://your-provider.com/auth/realms/your-realm`).                                                                           |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic account creation on successful authentication. Defaults to `true`.                                                                                                                             |

### Callback URL (Redirect URI)

Configure your OIDC provider with this callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` and `<OIDC_CLIENT_NAME>` with your settings.

## Translation

Help translate WYGIWYH at:

[https://translations.herculino.com/engage/wygiwyh/](https://translations.herculino.com/engage/wygiwyh/)

>   **Note:**  Login with your GitHub account.

## Caveats & Warnings

*   I am not an accountant. Review calculations and terms. Open an issue if you find any errors.
*   Most calculations are done at runtime, potentially affecting performance.
*   This is not a budgeting or double-entry accounting application. Consider other options if those features are essential.

## Built With

WYGIWYH is built with these awesome open-source tools:

*   Django
*   HTMX
*   _hyperscript
*   Procrastinate
*   Bootstrap
*   Tailwind
*   Webpack
*   PostgreSQL
*   Django REST framework
*   Alpine.js