# WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker

**WYGIWYH** (What You Get Is What You Have) empowers you to manage your money effectively with a no-budget, straightforward approach.  [Check out the original repo](https://github.com/eitchtee/WYGIWYH) for the source code and more!

## Key Features

*   **Unified Transaction Tracking:** Easily record all income and expenses in one place.
*   **Multi-Account Support:** Track your money across various accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:**  Manage transactions and balances in different currencies.
*   **Customizable Currencies:**  Define your own currencies for crypto, rewards points, and more.
*   **Automated Transaction Rules:** Set up rules to automatically adjust transactions.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:**  Track recurring investments for crypto and stocks.
*   **API Integration:** Seamlessly connect with other services for automated transaction synchronization.

## Why Choose WYGIWYH?

Tired of overly complex budgeting apps? WYGIWYH simplifies finance tracking by adhering to a simple principle: **Use what you earn this month for this month.** This allows you to focus on your cash flow without the constraints of traditional budgeting.

## Demo

Try out WYGIWYH!

*   **Demo Site:** [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)
*   **Credentials:**
    *   Email: `demo@demo.com`
    *   Password: `wygiwyhdemo`
*   **Important:**  Demo data is wiped regularly. Most automation features are disabled.

## Getting Started

WYGIWYH runs using Docker and Docker Compose.

1.  **Install Docker & Docker Compose:** Ensure you have Docker and docker-compose installed on your system. ([Docker Installation](https://docs.docker.com/engine/install/), [Docker Compose Installation](https://docs.docker.com/compose/install/))
2.  **Create a Project Directory (Optional):**
    ```bash
    mkdir WYGIWYH
    cd WYGIWYH
    ```
3.  **Create a `docker-compose.yml` file:**  Paste the contents of the example `docker-compose.prod.yml` file from the original repo and modify it according to your needs.
4.  **Create a `.env` file:**  Paste the contents of the `.env.example` file from the original repo and edit the variables.
5.  **Run the application:**
    ```bash
    docker compose up -d
    docker compose exec -it web python manage.py createsuperuser #Create Admin User
    ```

### Running Locally
When running locally:

1.  Remove `URL` from `.env`.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep default `DJANGO_ALLOWED_HOSTS` settings.
4.  Access from `localhost:OUTBOUND_PORT`.

### Environment Variables

Configure WYGIWYH with the following environment variables:

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                                                                        |
|-------------------------------|-------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DJANGO_ALLOWED_HOSTS`          | string      | localhost 127.0.0.1               | Allowed domains and IPs for the site. [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts).                                                                                                                                                         |
| `HTTPS_ENABLED`                 | true\|false | false                             | Enables secure cookies (HTTPS).                                                                                                                                                                                                                                                  |
| `URL`                           | string      | http://localhost http://127.0.0.1 | Trusted origins for POST requests.  [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins).                                                                                                                                                       |
| `SECRET_KEY`                    | string      | ""                                | Cryptographic signing key (must be unique).                                                                                                                                                                                                                                      |
| `DEBUG`                         | true\|false | false                             | Enables/disables debug mode (don't use in production).                                                                                                                                                                                                                             |
| `SQL_DATABASE`                  | string      | None *required                    | Postgres database name.                                                                                                                                                                                                                                                           |
| `SQL_USER`                      | string      | user                              | Postgres username.                                                                                                                                                                                                                                                                   |
| `SQL_PASSWORD`                | string      | password                          | Postgres password.                                                                                                                                                                                                                                                                   |
| `SQL_HOST`                      | string      | localhost                         | Postgres host address.                                                                                                                                                                                                                                                               |
| `SQL_PORT`                      | string      | 5432                              | Postgres port.                                                                                                                                                                                                                                                                      |
| `SESSION_EXPIRY_TIME`           | int         | 2678400 (31 days)                 | Session cookie lifetime (in seconds).                                                                                                                                                                                                                                        |
| `ENABLE_SOFT_DELETE`            | true\|false | false                             | Enables soft deletes for transactions (transactions are retained but marked as deleted).                                                                                                                                                                                         |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | 365                               | Days to keep soft-deleted transactions (0 for indefinite).  Only active if `ENABLE_SOFT_DELETE` is `true`.                                                                                                                                                                  |
| `TASK_WORKERS`                  | int         | 1                                 | Number of workers for asynchronous tasks.                                                                                                                                                                                                                                       |
| `DEMO`                          | true\|false | false                             | Enables demo mode (likely disables some features).                                                                                                                                                                                                                              |
| `ADMIN_EMAIL`                   | string      | None                              | Automatically creates an admin account (requires `ADMIN_PASSWORD` set).                                                                                                                                                                                                             |
| `ADMIN_PASSWORD`                | string      | None                              | Automatically creates an admin account (requires `ADMIN_EMAIL` set).                                                                                                                                                                                                             |
| `CHECK_FOR_UPDATES`             | bool        | true                              | Checks for new versions and notifies users.                                                                                                                                                                                                                                      |

## Unraid

WYGIWYH is available on the Unraid Store.  You'll need to set up your own PostgreSQL database (version 15+).

*   **Create Admin User:** Open the container's console in the Unraid UI (Docker page -> WYGIWYH icon -> Console), and run `python manage.py createsuperuser`.

## OIDC Configuration

Configure OIDC logins for single sign-on.

| Variable             | Description                                                                                                                                                                                                                                                            |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | Provider name (displayed on the login page). Defaults to `OpenID Connect`.                                                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | Your OIDC provider's Client ID.                                                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | Your OIDC provider's Client Secret.                                                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | Your OIDC provider's base URL (discovery document or authorization server URL).                                                                                                                                                                                        |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic account creation on successful authentication. Defaults to `true`.                                                                                                                                                                              |

**Callback URL (Redirect URI):** `https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/` (Replace `<OIDC_CLIENT_NAME>` with the slugified value of `OIDC_CLIENT_NAME` or `openid-connect`.)

## How It Works

For more detailed information, consult the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Help Translate WYGIWYH!

Contribute translations via:  [https://translations.herculino.com/engage/wygiwyh/](https://translations.herculino.com/engage/wygiwyh/) (Login with your GitHub account).

## Caveats and Warnings

*   Not an accounting expert; some calculations may need improvement.  Please report issues.
*   Calculations are done at runtime; performance might degrade with a large number of transactions.
*   Not a budgeting or double-entry accounting application.  Consider alternatives if you need these features.

## Built With

WYGIWYH is built on the following open source technologies:
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