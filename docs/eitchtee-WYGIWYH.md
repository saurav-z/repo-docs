<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Simplify Your Finances
  <br>
</h1>

<h4 align="center">Take control of your money with this powerful, no-budget finance tracker.</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#demo">Demo</a> •
  <a href="#environment-variables">Environment Variables</a> •
  <a href="#oidc-configuration">OIDC Configuration</a> •
  <a href="#help-us-translate">Help Translate</a> •
  <a href="#caveats-and-warnings">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** (pronounced "wiggy-wih") is a finance tracker built on the principle of *What You Get Is What You Have*, designed for straightforward money management without the complexities of budgeting. Track your income, expenses, and investments across multiple currencies with ease. 

![WYGIWYH Screenshots](./.github/img/monthly_view.png) ![WYGIWYH Screenshots](./.github/img/yearly.png) ![WYGIWYH Screenshots](./.github/img/networth.png) ![WYGIWYH Screenshots](./.github/img/calendar.png) ![WYGIWYH Screenshots](./.github/img/all_transactions.png)

## Key Features

*   **Unified Transaction Tracking:** Centralized view of all income and expenses.
*   **Multi-Account Support:** Track funds across various accounts (banks, wallets, investments).
*   **Multi-Currency Support:** Effortlessly manage transactions and balances in different currencies.
*   **Custom Currencies:** Create and track custom currencies for crypto, rewards, and more.
*   **Automated Adjustments:** Utilize customizable rules for automated transaction modifications.
*   **Dollar-Cost Averaging (DCA) Tracker:** Built-in tool for tracking recurring investments.
*   **API Support:** Integrate with other services to automate transaction synchronization.

## Why WYGIWYH?

Traditional budgeting can feel restrictive. WYGIWYH offers a simpler approach:

> Use what you earn this month for this month. Any savings are tracked but treated as untouchable for future months.

This straightforward principle lets you manage your finances without complicated budgets while still tracking your spending. Frustrated with existing solutions, WYGIWYH was built to offer the multi-currency, non-budgeting, and automation features that other apps lacked.

## Demo

Try WYGIWYH now with this demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

*   **Email:** `demo@demo.com`
*   **Password:** `wygiwyhdemo`

> [!NOTE]
> Data in the demo is reset frequently. Most advanced features are disabled.

## How to Use

WYGIWYH is built using Docker and Docker Compose. To get started:

1.  **Prerequisites:** Ensure you have [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed.
2.  **Clone and Configure:**
    ```bash
    # Create a folder (optional)
    $ mkdir WYGIWYH
    $ cd WYGIWYH
    $ touch docker-compose.yml
    $ nano docker-compose.yml
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs
    $ touch .env
    $ nano .env # or any other editor you want to use
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly
    ```
3.  **Run the Application:**
    ```bash
    $ docker compose up -d
    # Create the first admin account. This isn't required if you set the enviroment variables: ADMIN_EMAIL and ADMIN_PASSWORD.
    $ docker compose exec -it web python manage.py createsuperuser
    ```

> [!NOTE]
> For Unraid users, WYGIWYH is available in the Unraid Store. See the [Unraid Section](#unraid) and [Environment Variables](#environment-variables) for more details.

### Running Locally

To run locally:

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).
    Access the application at `localhost:OUTBOUND_PORT`.

> [!NOTE]
> Add your machine's IP to `DJANGO_ALLOWED_HOSTS` if running behind Tailscale or similar. Also add the IP if not using localhost.

### Latest Changes
Features are only added to `main` when ready, if you want to run the latest version, you must build from source or use the `:nightly` tag on docker. Keep in mind that there can be undocumented breaking changes.

All the required Dockerfiles are [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

## Unraid

WYGIWYH is also available on the Unraid store.  You'll need to provision your own Postgres database (version 15 or higher).

To create the first user, open the container's console using Unraid's UI, by clicking on WYGIWYH icon on the Docker page and selecting `Console`, then type `python manage.py createsuperuser`, you'll them be prompted to input your e-mail and password.

## Environment Variables

Customize WYGIWYH with these environment variables:

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                              |
|-------------------------------|-------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | Comma-separated list of domains and IPs for site access. [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                                                                             |
| HTTPS_ENABLED                 | true\|false | false                             | Enable secure cookies.                                                                                                                                                                                                                  |
| URL                           | string      | http://localhost http://127.0.0.1 | Comma-separated list of trusted origins for POST requests. [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                                                                        |
| SECRET_KEY                    | string      | ""                                | Unique secret key for cryptographic signing.                                                                                                                                                                                            |
| DEBUG                         | true\|false | false                             | Enable or disable debug mode. Don't use in production.                                                                                                                                                                                   |
| SQL_DATABASE                  | string      | None *required                    | Name of your Postgres database.                                                                                                                                                                                                           |
| SQL_USER                      | string      | user                              | Postgres database username.                                                                                                                                                                                                                |
| SQL_PASSWORD                | string      | password                          | Postgres database password.                                                                                                                                                                                                                |
| SQL_HOST                      | string      | localhost                         | Postgres database host address.                                                                                                                                                                                                           |
| SQL_PORT                      | string      | 5432                              | Postgres database port.                                                                                                                                                                                                                  |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                          |
| ENABLE_SOFT_DELETE            | true\|false | false                             | Enable soft deletion of transactions.                                                                                                                                                                                                    |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                               | Days to keep soft-deleted transactions. Only applies if `ENABLE_SOFT_DELETE` is true.                                                                                                                                                   |
| TASK_WORKERS                  | int         | 1                                 | Number of workers for async tasks.                                                                                                                                                                                                      |
| DEMO                          | true\|false | false                             | Enables demo mode.                                                                                                                                                                                                                        |
| ADMIN_EMAIL                   | string      | None                              | Email for automatic admin account creation. Requires `ADMIN_PASSWORD`.                                                                                                                                                                    |
| ADMIN_PASSWORD                | string      | None                              | Password for automatic admin account creation. Requires `ADMIN_EMAIL`.                                                                                                                                                                    |
| CHECK_FOR_UPDATES             | bool        | true                              | Check for and notify about new versions (checks GitHub API every 12 hours).                                                                                  |

## OIDC Configuration

Configure OpenID Connect (OIDC) login with these variables:

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | Provider name for the login page. Defaults to `OpenID Connect`.                                                                                                                                                                                           |
| `OIDC_CLIENT_ID`     | Client ID from your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | Client Secret from your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | Base URL for your OIDC provider's discovery document or authorization server.  `django-allauth` uses this to find endpoints.                                                                                                 |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of accounts on successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL:**

Configure your OIDC provider with this callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` and `<OIDC_CLIENT_NAME>` with your actual values.

## Help us Translate

Contribute to WYGIWYH translations:

[![Translation Status](https://translations.herculino.com/widget/wygiwyh/open-graph.png)](https://translations.herculino.com/engage/wygiwyh/)

> [!NOTE]
> Login with your GitHub account.

## Caveats and Warnings

*   I'm not an accountant, and some calculations may have errors. Please open an issue if you find any.
*   Calculations are performed at runtime, which can affect performance.
*   This is not a budgeting or double-entry accounting application.

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