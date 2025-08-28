<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Simple & Powerful Finance Tracker
  <br>
</h1>

<h4 align="center">Take control of your finances with a no-budget, principles-first approach.</h4>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#translate">Translate</a> •
  <a href="#caveats">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH (What You Get Is What You Have)** is a finance tracker built for straightforward money management, offering features like multi-currency support and a built-in dollar-cost averaging tracker to help you understand and manage your finances.

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## Overview

WYGIWYH (pronounced "wiggy-wih") simplifies finance tracking with a simple principle: Use what you earn this month for this month.  Built with multi-currency support, custom transactions, and an API for automation.  It helps you avoid budgeting while staying informed about your spending. Read more in the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Key Features

*   **Comprehensive Transaction Tracking:** Record all income and expenses in one place.
*   **Multiple Account Support:** Track various accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:** Manage transactions and balances in multiple currencies.
*   **Custom Currencies:** Create custom currencies for various tracking needs.
*   **Automated Adjustments:** Apply custom rules to automatically modify transactions.
*   **Dollar-Cost Average (DCA) Tracker:** Essential for tracking recurring investments.
*   **API Support:** Integrate seamlessly with other services for transaction automation.

## Demo

Experience WYGIWYH firsthand: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

> [!NOTE]
> *   Email: `demo@demo.com`
> *   Password: `wygiwyhdemo`
>
>   Any data added will be wiped in 24 hours or less, and most automation features are disabled.

## How to Use

WYGIWYH is deployed using Docker.

**Prerequisites:** Docker and Docker Compose installed.

```bash
# Create a project folder (optional)
mkdir WYGIWYH
cd WYGIWYH

# Create and edit your docker-compose file.
touch docker-compose.yml
nano docker-compose.yml
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs

# Create and edit your .env file
touch .env
nano .env
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly

# Run the application
docker compose up -d

# Create the first admin account if the env variables are not set
docker compose exec -it web python manage.py createsuperuser
```

> [!NOTE]
> If you're using Unraid, use the app on the store, and reference the [Unraid section](#unraid) for further guidance.

### Running Locally

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

Access locally via `localhost:OUTBOUND_PORT`.

> [!NOTE]
> *   Add your machine IP to `DJANGO_ALLOWED_HOSTS` if using Tailscale or similar services.
> *   For IPs other than localhost, include in `DJANGO_ALLOWED_HOSTS` without `http://`.

### Latest Changes

The `main` branch holds the stable release. Run the `:nightly` tag on Docker for the latest version.

All the required Dockerfiles are [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

### Unraid

[nwithan8](https://github.com/nwithan8) has created a Unraid template for WYGIWYH, available in the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is available in the Unraid Store. You must provision your own Postgres (version 15 or higher) database.

To create the initial user, open the container's console via the Unraid UI, then type `python manage.py createsuperuser`.

## Environment Variables

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                                   |
|-------------------------------|-------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DJANGO_ALLOWED_HOSTS`          | string      | localhost 127.0.0.1               | Space-separated list of allowed domains and IPs. [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                                                                                     |
| `HTTPS_ENABLED`                 | true\|false | false                             | Enables/disables secure cookies.                                                                                                                                                                                                            |
| `URL`                           | string      | http://localhost http://127.0.0.1 | Space-separated list of trusted origins. [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                                                                                       |
| `SECRET_KEY`                    | string      | ""                                | Unique secret key for cryptographic signing.                                                                                                                                                                                             |
| `DEBUG`                         | true\|false | false                             | Enables/disables debug mode.  Do not use in production.                                                                                                                                                                                   |
| `SQL_DATABASE`                  | string      | None *required                    | Postgres database name.                                                                                                                                                                                                                     |
| `SQL_USER`                      | string      | user                              | Postgres username.                                                                                                                                                                                                                            |
| `SQL_PASSWORD`                | string      | password                          | Postgres password.                                                                                                                                                                                                                            |
| `SQL_HOST`                      | string      | localhost                         | Postgres host address.                                                                                                                                                                                                                      |
| `SQL_PORT`                      | string      | 5432                              | Postgres port.                                                                                                                                                                                                                              |
| `SESSION_EXPIRY_TIME`           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                            |
| `ENABLE_SOFT_DELETE`            | true\|false | false                             | Enables/disables soft delete for transactions.                                                                                                                                                                                            |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | 365                               | Time (days) to keep soft-deleted transactions.  0 to keep indefinitely.  Only works if `ENABLE_SOFT_DELETE` is true.                                                                                                                        |
| `TASK_WORKERS`                  | int         | 1                                 | Number of workers for async tasks.                                                                                                                                                                                                        |
| `DEMO`                          | true\|false | false                             | Enables demo mode.                                                                                                                                                                                                                            |
| `ADMIN_EMAIL`                   | string      | None                              | Creates an admin account automatically if `ADMIN_PASSWORD` is also set.                                                                                                                                                                     |
| `ADMIN_PASSWORD`                | string      | None                              | Creates an admin account automatically if `ADMIN_EMAIL` is also set.                                                                                                                                                                        |
| `CHECK_FOR_UPDATES`             | bool        | true                              | Checks for new versions and notifies users.                                                                                                                                                                                                  |

## OIDC Configuration

WYGIWYH supports login via OpenID Connect (OIDC) using `django-allauth`.

> [!NOTE]
> Currently only OpenID Connect is supported as a provider.

Configure OIDC with these environment variables:

| Variable             | Description                                                                                                                                                                                                                                                                                                                |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | Provider name displayed in the login page. Defaults to `OpenID Connect`.                                                                                                                                                                                                                                                    |
| `OIDC_CLIENT_ID`     | Client ID from your OIDC provider.                                                                                                                                                                                                                                                                                        |
| `OIDC_CLIENT_SECRET` | Client Secret from your OIDC provider.                                                                                                                                                                                                                                                                                    |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.).                                                           |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                                                                                                    |

**Callback URL (Redirect URI):**

Set the following callback URL in your OIDC provider configuration:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH instance URL and `<OIDC_CLIENT_NAME>` with the slugified `OIDC_CLIENT_NAME` (or `openid-connect` if unspecified).

## How it Works

For detailed information, check out the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Translate

Help translate WYGIWYH!  Contribute via [Herculino Translations](https://translations.herculino.com/engage/wygiwyh/).

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your GitHub account.

## Caveats and Warnings

*   I'm not an accountant, so terms and calculations may have errors. Please report issues.
*   Most calculations are done at runtime, which can affect performance.
*   Not a budgeting or double-entry accounting app. Open a discussion if these are desired.

## Built With

WYGIWYH leverages these amazing open-source tools:

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