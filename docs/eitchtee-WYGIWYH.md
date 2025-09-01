<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Simplified Finance Tracker
  <br>
</h1>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#caveats">Caveats</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** simplifies personal finance management with a "What You Get Is What You Have" approach, empowering you to track income, expenses, and investments without complex budgeting.

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## Overview

WYGIWYH (pronounced "wiggy-wih") offers a straightforward, no-budget approach to finance tracking.  Focus on using what you earn each month, tracking savings separately.  Ideal for users seeking multi-currency support, flexible transaction management, and a clean, simple interface. This finance tracker is built for those seeking a simple, flexible, and automated way to manage their finances.

## Key Features

*   **Unified Transaction Tracking:** Easily record all income and expenses in one place.
*   **Multi-Account Support:** Track finances across multiple accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:** Manage transactions and balances in various currencies.
*   **Custom Currencies:** Create your own currencies for crypto, rewards points, or other unique models.
*   **Automated Transaction Rules:** Automatically adjust transactions with customizable rules.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Simplify tracking recurring investments (crypto, stocks).
*   **API Support:** Integrate WYGIWYH with other services for automation.

## Demo

Experience WYGIWYH firsthand!

[wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

**Demo Credentials:**

*   Email: `demo@demo.com`
*   Password: `wygiwyhdemo`

**Note:** Data added to the demo is wiped daily. Most automation features are disabled in the demo.

## Getting Started

To run WYGIWYH, you'll need [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).

1.  **Create a Project Directory (optional):**
    ```bash
    mkdir WYGIWYH
    cd WYGIWYH
    ```
2.  **Create docker-compose.yml:**
    ```bash
    touch docker-compose.yml
    nano docker-compose.yml
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs
    ```
3.  **Configure .env:**
    ```bash
    touch .env
    nano .env # or any other editor you want to use
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly
    ```
4.  **Run the Application:**
    ```bash
    docker compose up -d
    ```
5.  **Create the first admin account (if using default settings):**
    ```bash
    docker compose exec -it web python manage.py createsuperuser
    ```

For local development, remove `URL` from the `.env` file, set `HTTPS_ENABLED` to `false`, and use `localhost:OUTBOUND_PORT` for access.  If running behind a service like Tailscale, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.

### Unraid Installation

WYGIWYH is also available on the Unraid Store. You'll need to provision your own PostgreSQL database. To create your first user, open the container's console and run `python manage.py createsuperuser`.

### Environment Variables

Customize WYGIWYH with the following environment variables:

| Variable                      | Type        | Default                  | Description                                                                                                                                                                                                            |
|-------------------------------|-------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DJANGO_ALLOWED_HOSTS`        | string      | localhost 127.0.0.1     | Domains and IPs that WYGIWYH can serve (space-separated).                                                                                                                                                             |
| `HTTPS_ENABLED`               | true\|false | false                   | Enables secure cookies (true sets the 'secure' flag).                                                                                                                                                                  |
| `URL`                         | string      | http://localhost        | Trusted origins for unsafe requests (e.g., POST).                                                                                                                                                                   |
| `SECRET_KEY`                  | string      | ""                       |  Unique, unpredictable value for cryptographic signing.                                                                                                                                                                  |
| `DEBUG`                       | true\|false | false                   | Enables/disables debug mode.  Do not use in production.                                                                                                                                                              |
| SQL_DATABASE                  | string      | None                     | The name of your postgres database                                                                                                                                                                                         |
| SQL_USER                      | string      | user                     | The username used to connect to your postgres database                                                                                                                                                                   |
| SQL_PASSWORD                  | string      | password                 | The password used to connect to your postgres database                                                                                                                                                                   |
| SQL_HOST                      | string      | localhost                | The address used to connect to your postgres database                                                                                                                                                                    |
| SQL_PORT                      | string      | 5432                     | The port used to connect to your postgres database                                                                                                                                                                       |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)       | Session cookie lifetime in seconds.                                                                                                                                                                                    |
| ENABLE_SOFT_DELETE            | true\|false | false                   | Enables soft deleting of transactions.                                                                                                                                                                                 |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                      | Days to keep soft-deleted transactions (0 for indefinitely).  Requires `ENABLE_SOFT_DELETE`.                                                                                                                         |
| TASK_WORKERS                  | int         | 1                        | Number of workers for asynchronous tasks.                                                                                                                                                                              |
| DEMO                          | true\|false | false                   | Enables demo mode (limited functionality).                                                                                                                                                                             |
| ADMIN_EMAIL                   | string      | None                     | Automatically create an admin account with this email (requires `ADMIN_PASSWORD`).                                                                                                                                   |
| ADMIN_PASSWORD                | string      | None                     | Automatically create an admin account with this password (requires `ADMIN_EMAIL`).                                                                                                                                   |
| CHECK_FOR_UPDATES             | bool        | true                     | Check for updates and notify users (checks GitHub API every 12 hours).   |

### OIDC Configuration

Configure OpenID Connect (OIDC) login:

| Variable             | Description                                                                                                                                                                                                                                                              |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider (displayed on the login page).  Defaults to `OpenID Connect`.                                                                                                                                                                                  |
| `OIDC_CLIENT_ID`     | Client ID from your OIDC provider.                                                                                                                                                                                                                                      |
| `OIDC_CLIENT_SECRET` | Client Secret from your OIDC provider.                                                                                                                                                                                                                                  |
| `OIDC_SERVER_URL`    | Base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`).                                                                                                                                     |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic creation of new accounts on successful authentication.  Defaults to `true`.                                                                                                                                                                            |

**Callback URL:**

When configuring your OIDC provider, use the following callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH instance's URL and `<OIDC_CLIENT_NAME>` with your OIDC_CLIENT_NAME value.  If this variable has not been configured then it defaults to `openid-connect`.

## Contributing

Help improve WYGIWYH!  For details, see the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Caveats

*   Calculations are performed at runtime, potentially affecting performance.
*   This is not a budgeting or double-entry accounting application.

## Built With

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

[Back to Top](#wygiwyh-your-simplified-finance-tracker)