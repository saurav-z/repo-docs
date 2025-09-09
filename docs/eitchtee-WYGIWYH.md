<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a No-Budget Approach
  <br>
</h1>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Live Demo</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#translation">Translation</a> •
  <a href="#caveats">Caveats and Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH (What You Get Is What You Have)** is a powerful and opinionated finance tracker designed for those who prefer a simple, no-budget approach to money management.  It emphasizes tracking income and expenses within the current month, without the constraints of traditional budgeting.

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## Introduction

Tired of complex budgeting apps? WYGIWYH offers a refreshing alternative. This finance tracker operates on a straightforward principle: **Use what you earn this month for this month.**  This means savings are tracked but treated as untouchable for future months. WYGIWYH provides the tools you need to effectively manage your money without the complexities of traditional budgeting.

## Key Features

WYGIWYH simplifies personal finance tracking with these key features:

*   **Unified Transaction Tracking:**  Record all income and expenses in one place.
*   **Multi-Account Support:** Track money and assets across various accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:** Manage transactions and balances in different currencies out-of-the-box.
*   **Custom Currencies:**  Create custom currencies for crypto, rewards points, or other models.
*   **Automated Adjustments with Rules:** Automatically modify transactions using customizable rules.
*   **Built-in Dollar-Cost Average (DCA) Tracker:** Track recurring investments, especially for crypto and stocks.
*   **API Support for Automation:** Seamlessly integrate with existing services to synchronize transactions.

## Demo

Experience WYGIWYH firsthand with our live demo:

*   **Demo URL:** [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)
*   **Credentials:**
    *   Email: `demo@demo.com`
    *   Password: `wygiwyhdemo`

> **Important:**  Data added to the demo is reset regularly (within 24 hours).  Most automation features (API, Rules, Exchange Rates, Import/Export) are disabled in the demo.

## Getting Started

WYGIWYH runs using Docker and Docker Compose.

1.  **Prerequisites:** Ensure you have [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed.
2.  **Configuration:**
    *   Create a `docker-compose.yml` file using the example provided in the [repository](https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml)
    *   Create a `.env` file using the example provided in the [repository](https://github.com/eitchtee/WYGIWYH/blob/main/.env.example) and configure your settings.
3.  **Run the App:** Execute `docker compose up -d` in your terminal.
4.  **Create Admin User:** Run `docker compose exec -it web python manage.py createsuperuser` to create the initial admin account. You can skip this step if you set `ADMIN_EMAIL` and `ADMIN_PASSWORD` environment variables.

### Running Locally

To run locally:

1.  In your `.env` file:
    *   Remove `URL`
    *   Set `HTTPS_ENABLED` to `false`
    *   Leave the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1])
2.  Access the application via `localhost:OUTBOUND_PORT` (check your `docker-compose.yml` for the correct port).

### Latest Changes

For the latest features, consider building from source or using the `:nightly` tag on Docker. Be aware that there may be undocumented breaking changes in nightly builds.

### Unraid

WYGIWYH is available on the Unraid Store. You'll need to provision your own PostgreSQL (version 15 or up) database.

1.  Open the container's console via the Unraid UI (Docker page -> WYGIWYH icon -> Console).
2.  Run `python manage.py createsuperuser` and follow the prompts to create an admin user.

## Configuration

### Environment Variables

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                |
| ----------------------------- | ----------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | Space-separated list of allowed domains and IPs for the site. [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                          |
| HTTPS_ENABLED                 | true\|false | false                             | Enables secure cookies (HTTPS).                                                                                                                                                             |
| URL                           | string      | http://localhost http://127.0.0.1 | List of trusted origins for unsafe requests (e.g., POST). [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                       |
| SECRET_KEY                    | string      | ""                                | Cryptographic signing key; must be unique and unpredictable.                                                                                                                                 |
| DEBUG                         | true\|false | false                             | Enables/disables debug mode (do not use in production).                                                                                                                                       |
| SQL_DATABASE                  | string      | None *required                    | Your PostgreSQL database name.                                                                                                                                                               |
| SQL_USER                      | string      | user                              | PostgreSQL username.                                                                                                                                                                         |
| SQL_PASSWORD                  | string      | password                          | PostgreSQL password.                                                                                                                                                                         |
| SQL_HOST                      | string      | localhost                         | PostgreSQL host address.                                                                                                                                                                     |
| SQL_PORT                      | string      | 5432                              | PostgreSQL port.                                                                                                                                                                           |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                            |
| ENABLE_SOFT_DELETE            | true\|false | false                             | Enables soft-deleting transactions (deleted transactions remain in the database).                                                                                                            |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                               | Days to keep soft-deleted transactions (only applies if `ENABLE_SOFT_DELETE` is true).  0 = indefinitely.                                                                                  |
| TASK_WORKERS                  | int         | 1                                 | Number of workers for asynchronous tasks.                                                                                                                                                     |
| DEMO                          | true\|false | false                             | Enables demo mode.                                                                                                                                                                           |
| ADMIN_EMAIL                   | string      | None                              | Automatically creates an admin account with this email (requires `ADMIN_PASSWORD`).                                                                                                           |
| ADMIN_PASSWORD                | string      | None                              | Automatically creates an admin account with this password (requires `ADMIN_EMAIL`).                                                                                                          |
| CHECK_FOR_UPDATES             | bool        | true                              | Checks for new versions and notifies users. Checks Github's API every 12 hours.                                                                                                                 |

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) for user authentication via `django-allauth`.

| Variable             | Description                                                                                                                                                                                                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | Provider name (displayed in login). Defaults to `OpenID Connect`.                                                                                                                                                                                         |
| `OIDC_CLIENT_ID`     | Client ID from your OIDC provider.                                                                                                                                                                                                                       |
| `OIDC_CLIENT_SECRET` | Client Secret from your OIDC provider.                                                                                                                                                                                                                   |
| `OIDC_SERVER_URL`    | Base URL of your OIDC provider's discovery document/authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`).                                                                                                                 |
| `OIDC_ALLOW_SIGNUP`  | Allows automatic creation of accounts on successful authentication. Defaults to `true`.                                                                                                                                                                 |

**Callback URL (Redirect URI):**

The default callback URL for your OIDC provider is:
`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH instance's URL and `<OIDC_CLIENT_NAME>` with your OIDC_CLIENT_NAME or `openid-connect`.

## How It Works

For detailed information on how WYGIWYH operates, consult the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Translation

Help translate WYGIWYH via [Herculino Translations](https://translations.herculino.com/engage/wygiwyh/).  Login with your GitHub account.

## Caveats and Warnings

*   This is not a budgeting or double-entry accounting application.
*   Some calculations might be inaccurate – open an issue if you find any.
*   Most calculations are done at runtime, which can impact performance.  (Expect 500ms load times on pages with many transactions and exchange rates).

## Built With

WYGIWYH is built with:

*   Django
*   HTMX
*   \_hyperscript
*   Procrastinate
*   Bootstrap
*   Tailwind
*   Webpack
*   PostgreSQL
*   Django REST framework
*   Alpine.js