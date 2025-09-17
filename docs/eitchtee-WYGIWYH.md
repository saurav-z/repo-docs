<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker
  <br>
</h1>

<p align="center">
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#environment-variables">Environment Variables</a> •
  <a href="#oidc-configuration">OIDC Configuration</a> •
  <a href="#how-it-works">How it Works</a> •
  <a href="#help-us-translate-wygiwyh">Translate</a> •
  <a href="#caveats-and-warnings">Caveats and Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a straightforward and feature-rich finance tracker designed for those who prefer a no-budget approach to managing their money.  It's perfect for anyone seeking a simple yet powerful way to track income, expenses, and investments.

[![WYGIWYH Screenshots](.github/img/monthly_view.png)](https://github.com/eitchtee/WYGIWYH)
[![WYGIWYH Screenshots](.github/img/yearly.png)](https://github.com/eitchtee/WYGIWYH)
[![WYGIWYH Screenshots](.github/img/networth.png)](https://github.com/eitchtee/WYGIWYH)
[![WYGIWYH Screenshots](.github/img/calendar.png)](https://github.com/eitchtee/WYGIWYH)
[![WYGIWYH Screenshots](.github/img/all_transactions.png)](https://github.com/eitchtee/WYGIWYH)

## Why WYGIWYH?

Managing your finances shouldn't be complex. WYGIWYH (pronounced "wiggy-wih") operates on a simple principle:

> Use what you earn this month for this month. Savings are tracked but treated as untouchable for future months.

This approach helps you avoid overspending while providing clear visibility into your financial health.

Driven by the need for a tool that met specific requirements – including multi-currency support, no budgeting constraints, web app usability, API integration, and custom transaction rules – WYGIWYH was created.

## Key Features

**WYGIWYH** simplifies personal finance tracking with these key features:

*   **Unified Transaction Tracking:**  Organize all income and expenses in one place.
*   **Multi-Account Support:** Track funds across banks, wallets, and investment accounts.
*   **Multi-Currency Support:** Manage transactions and balances in various currencies.
*   **Custom Currency Creation:**  Define currencies for crypto, rewards points, or other needs.
*   **Automated Rules:** Automatically modify transactions with customizable rules.
*   **Dollar-Cost Averaging (DCA) Tracker:**  Track recurring investments (crypto, stocks, etc.).
*   **API Support:**  Integrate with other services for automated transaction synchronization.

## Demo

Experience WYGIWYH firsthand with our demo:  [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

> [!NOTE]
> Email: `demo@demo.com`
> Password: `wygiwyhdemo`

Please note that demo data is reset frequently.  Most automation features are disabled in the demo.

## How To Use

WYGIWYH requires [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).

1.  **Create a project directory:**
    ```bash
    mkdir WYGIWYH
    cd WYGIWYH
    ```
2.  **Create `docker-compose.yml`:**
    ```bash
    touch docker-compose.yml
    nano docker-compose.yml
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs
    ```
3.  **Create `.env` file:**
    ```bash
    touch .env
    nano .env # or any other editor you want to use
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly
    ```
4.  **Run the application:**
    ```bash
    docker compose up -d
    ```
5.  **Create the first admin account:**  (Optional if `ADMIN_EMAIL` and `ADMIN_PASSWORD` are set)
    ```bash
    docker compose exec -it web python manage.py createsuperuser
    ```

> [!NOTE]
> If you're using Unraid, see the [Unraid section](#unraid) and [Environment Variables](#environment-variables) for details.

### Running Locally

To run WYGIWYH locally:

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

Access the application at `localhost:OUTBOUND_PORT`.

> [!NOTE]
> -   If using Tailscale or similar, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
> -   For non-localhost IPs, add them to `DJANGO_ALLOWED_HOSTS`.

### Latest Changes

Features are added to `main` when they are ready.  To get the latest version, build from source or use the `:nightly` tag with Docker. Be aware of potential undocumented breaking changes.

Find the Dockerfiles [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

## Unraid

[nwithan8](https://github.com/nwithan8) provides a Unraid template for WYGIWYH. See the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is also available in the Unraid Store. You'll need your own PostgreSQL (version 15+) database.

Create the first user via the Unraid UI console (click the Docker icon, select `Console`, then type `python manage.py createsuperuser`).

## Environment Variables

Customize WYGIWYH with the following environment variables:

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                              |
|-------------------------------|-------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | Space-separated list of domains and IPs where the site is served.  [Learn more](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts).                                                                                                 |
| HTTPS_ENABLED                 | true\|false | false                             | Enables secure cookies.                                                                                                                                                                                                                              |
| URL                           | string      | http://localhost http://127.0.0.1 | Space-separated list of trusted origins for unsafe requests (e.g., POST).  [Learn more](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins).                                                                 |
| SECRET_KEY                    | string      | ""                                | A unique, unpredictable value for cryptographic signing.                                                                                                                                                                      |
| DEBUG                         | true\|false | false                             | Enable/disable debug mode (don't use in production).                                                                                                                                                                           |
| SQL_DATABASE                  | string      | None *required                    | Your PostgreSQL database name.                                                                                                                                                                                                       |
| SQL_USER                      | string      | user                              | PostgreSQL username.                                                                                                                                                                                                                  |
| SQL_PASSWORD                  | string      | password                          | PostgreSQL password.                                                                                                                                                                                                                  |
| SQL_HOST                      | string      | localhost                         | PostgreSQL host address.                                                                                                                                                                                                                   |
| SQL_PORT                      | string      | 5432                              | PostgreSQL port.                                                                                                                                                                                                                      |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                                          |
| ENABLE_SOFT_DELETE            | true\|false | false                             | Enable/disable soft deletes for transactions.                                                                                                                                                                         |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                               | Time (days) to keep soft-deleted transactions (only works if `ENABLE_SOFT_DELETE` is true). 0 to keep indefinitely.                                                                                                                                                            |
| TASK_WORKERS                  | int         | 1                                 | Number of workers for async tasks.                                                                                                                                                                                                                      |
| DEMO                          | true\|false | false                             | Enable/disable demo mode.                                                                                                                                                                                                                 |
| ADMIN_EMAIL                   | string      | None                              | Automatically creates an admin account with this email.  Requires `ADMIN_PASSWORD`.                                                                                                                                             |
| ADMIN_PASSWORD                | string      | None                              | Automatically creates an admin account with this password. Requires `ADMIN_EMAIL`.                                                                                                                                             |
| CHECK_FOR_UPDATES             | bool        | true                              | Check for and notify about new versions (checks GitHub API every 12 hours).                                                                                 |

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) authentication through `django-allauth`.

> [!NOTE]
> Only OpenID Connect is currently supported.

Configure OIDC using these environment variables:

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | Provider name (displayed in login page).  Defaults to `OpenID Connect`.                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | Client ID from your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | Client Secret from your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | OIDC provider's discovery document or authorization server URL (e.g., `https://your-provider.com/auth/realms/your-realm`).  `django-allauth` discovers endpoints from here.                                                               |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic account creation on successful authentication.  Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

Configure your OIDC provider with this callback URL (replace the domain with your instance's URL):

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`
`<OIDC_CLIENT_NAME>` is the slugified name of `OIDC_CLIENT_NAME`, or `openid-connect` if not set.

## How it Works

See the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki) for detailed information.

## Help us translate WYGIWYH!

Contribute to WYGIWYH translations:
<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your GitHub account

## Caveats and Warnings

*   I'm not an accountant, and some calculations may have errors. Please open an issue if you find any.
*   Most calculations are done at runtime, which may affect performance.
*   WYGIWYH is not a budgeting or double-entry accounting application.

## Built With

WYGIWYH is built with these open-source tools:

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