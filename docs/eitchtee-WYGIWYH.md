<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple Approach
  <br>
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#environment-variables">Environment Variables</a> •
  <a href="#oidc-configuration">OIDC Configuration</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#translation">Translation</a> •
  <a href="#caveats">Caveats and Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH (What You Get Is What You Have)** is a powerful, open-source finance tracker designed for straightforward money management, focusing on simplicity and flexibility.  This finance tracker helps you visualize your spending habits, track your investments, and manage multiple currencies.  Manage your money with ease, using a no-budget, principles-first approach.

<img src=".github/img/monthly_view.png" width="18%"></img> <img src=".github/img/yearly.png" width="18%"></img> <img src=".github/img/networth.png" width="18%"></img> <img src=".github/img/calendar.png" width="18%"></img> <img src=".github/img/all_transactions.png" width="18%"></img>

## About

WYGIWYH offers a refreshing alternative to traditional budgeting apps. Inspired by the philosophy of using what you earn each month for that month, it simplifies financial tracking by focusing on income, expenses, and savings, without the constraints of budgeting. This approach simplifies financial management, giving you a clearer picture of your financial health.

## Key Features

*   **Unified Transaction Tracking:** Easily record all income and expenses in one place.
*   **Multi-Account Support:** Keep track of funds across various accounts (banks, wallets, investments, etc.).
*   **Built-in Multi-Currency Support:** Manage transactions and balances in multiple currencies seamlessly.
*   **Custom Currencies:** Create your own currencies for tracking things like crypto or rewards points.
*   **Automated Adjustments with Rules:** Customize transaction rules for automated modifications.
*   **Dollar-Cost Averaging (DCA) Tracker:** Track recurring investments, especially for crypto and stocks.
*   **API Support:** Integrate with other services to automate transaction synchronization.

## Demo

Explore WYGIWYH's capabilities with a demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

**Demo Credentials:**

>   Email: `demo@demo.com`
>   Password: `wygiwyhdemo`

**Important:** Data added to the demo is reset regularly.

## Getting Started

WYGIWYH is easily deployable using Docker and docker-compose.

1.  **Prerequisites:** Ensure you have [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed.
2.  **Configuration:**
    *   Create a `.env` file based on the provided `.env.example` to configure your setup.
    *   Use the provided `docker-compose.prod.yml` file.
3.  **Deployment:** Run `docker compose up -d` from your command line to launch the application.
4.  **Initial Setup:** Create an admin account using `docker compose exec -it web python manage.py createsuperuser`.  Alternatively, set `ADMIN_EMAIL` and `ADMIN_PASSWORD` in your `.env` file for automatic admin account creation.

### Running Locally

For local development, modify your `.env` file:

1.  Remove the `URL` setting.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

You can then access the application via `localhost:OUTBOUND_PORT`.  If you are using Tailscale or another similar service, add the IP address of your machine to `DJANGO_ALLOWED_HOSTS`.

## Environment Variables

Configure WYGIWYH with these environment variables:

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                                                          |
|-------------------------------|-------------|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DJANGO_ALLOWED_HOSTS`          | string      | localhost 127.0.0.1               | List of space-separated domains and IPs for your WYGIWYH site.  See [Django documentation](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts).                                                                                               |
| `HTTPS_ENABLED`                 | true/false  | false                             | Enables secure cookies.  Set to `true` for HTTPS.                                                                                                                                                                                                             |
| `URL`                           | string      | http://localhost http://127.0.0.1 | List of space-separated domains and IPs (with protocol) representing trusted origins for unsafe requests. See [Django documentation](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins).                                            |
| `SECRET_KEY`                    | string      | ""                                | A unique, unpredictable value for cryptographic signing.                                                                                                                                                                                                        |
| `DEBUG`                         | true/false  | false                             | Turns DEBUG mode on/off. Do not use in production.                                                                                                                                                                                                           |
| `SQL_DATABASE`                  | string      | None *required                    | The name of your PostgreSQL database.                                                                                                                                                                                                                            |
| `SQL_USER`                      | string      | user                              | The username for your PostgreSQL database.                                                                                                                                                                                                                        |
| `SQL_PASSWORD`                | string      | password                          | The password for your PostgreSQL database.                                                                                                                                                                                                                        |
| `SQL_HOST`                      | string      | localhost                         | The address for your PostgreSQL database.                                                                                                                                                                                                                         |
| `SQL_PORT`                      | string      | 5432                              | The port for your PostgreSQL database.                                                                                                                                                                                                                            |
| `SESSION_EXPIRY_TIME`           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                                                  |
| `ENABLE_SOFT_DELETE`            | true/false  | false                             | Enables soft deletes for transactions.                                                                                                                                                                                                                          |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | 365                               | Days to keep soft-deleted transactions.  Only works if `ENABLE_SOFT_DELETE` is true.  If set to 0, transactions are kept indefinitely.                                                                                                                  |
| `TASK_WORKERS`                  | int         | 1                                 | Number of workers for asynchronous tasks.                                                                                                                                                                                                                       |
| `DEMO`                          | true/false  | false                             | Enables demo mode.                                                                                                                                                                                                                                                |
| `ADMIN_EMAIL`                   | string      | None                              | Automatically creates an admin account with this email if `ADMIN_PASSWORD` is also set.                                                                                                                                                                        |
| `ADMIN_PASSWORD`                | string      | None                              | Automatically creates an admin account with this password if `ADMIN_EMAIL` is also set.                                                                                                                                                                       |
| `CHECK_FOR_UPDATES`             | bool        | true                              | Checks for and notifies users of new versions (queries GitHub API every 12 hours).                                                                                                                                                                                |

## OIDC Configuration

Integrate with OpenID Connect (OIDC) providers for user authentication using `django-allauth`.

| Variable             | Description                                                                                                                                                                                                 |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | Provider name (displayed on login page), defaults to `OpenID Connect`.                                                                                                                                   |
| `OIDC_CLIENT_ID`     | Your OIDC provider's Client ID.                                                                                                                                                                           |
| `OIDC_CLIENT_SECRET` | Your OIDC provider's Client Secret.                                                                                                                                                                       |
| `OIDC_SERVER_URL`    | OIDC provider's discovery document or authorization server base URL.  `django-allauth` uses this to discover endpoints.                                                                                    |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic account creation on successful authentication, defaults to `true`.                                                                                                                       |

**Callback URL (Redirect URI):**

Set your OIDC provider's callback URL to:
`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your actual WYGIWYH URL and `<OIDC_CLIENT_NAME>` with the slugified `OIDC_CLIENT_NAME` or `openid-connect` if not set.

## How It Works

For detailed information, please refer to the [WYGIWYH Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Translation

Help translate WYGIWYH via:  <a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

>   Log in with your GitHub account.

## Caveats and Warnings

*   I am not an accountant; some terms and calculations may be inaccurate. Please open an issue if you find anything that needs improvement.
*   Most calculations are done at runtime, which can impact performance.  (Load times average around 500ms on my personal instance with 3000+ transactions and 4000+ exchange rates).
*   This application is not a budgeting or double-entry accounting tool.  If you require these features, there are other options available.

## Built With

WYGIWYH leverages these open-source tools:

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