<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances
  <br>
</h1>

<h4 align="center">A powerful, opinionated finance tracker built on simplicity.</h4>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#how-it-works">How it Works</a> •
  <a href="#translation">Translation</a> •
  <a href="#caveats-and-warnings">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a finance tracker designed for a straightforward, no-budget approach to money management.  Focus on simplicity and flexibility to track your income, expenses, and investments.

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## Introduction

Simplify your finances with **WYGIWYH**, a finance tracker designed for those who prefer a no-budget approach. WYGIWYH is based on a simple principle: use what you earn this month for this month. Savings are tracked but treated as untouchable for future months.

Frustrated by the lack of tools that met the requirements, WYGIWYH was born:

1.  Multi-currency support.
2.  No budgeting constraints.
3.  Web app usability (with mobile support).
4.  API for automation.
5.  Custom transaction rules.

## Key Features

WYGIWYH offers comprehensive features to streamline your personal finance tracking:

*   **Unified Transaction Tracking:** Record all income and expenses in one place.
*   **Multiple Account Support:** Track your money and assets (banks, wallets, investments, etc.).
*   **Multi-Currency Support:** Manage transactions and balances in different currencies.
*   **Custom Currencies:** Create your own currencies for crypto, rewards points, or any models.
*   **Automated Adjustments with Rules:** Automatically modify transactions using customizable rules.
*   **Built-in Dollar-Cost Average (DCA) Tracker:** Track recurring investments.
*   **API Support for Automation:** Seamlessly integrate with existing services.

## Demo

Try out WYGIWYH with the demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

>   E-mail: `demo@demo.com`
>
>   Password: `wygiwyhdemo`

**Important:** Data added will be wiped within 24 hours. Automation features are disabled.

## How To Use

To run this application, you'll need [Docker](https://docs.docker.com/engine/install/) with [docker-compose](https://docs.docker.com/compose/install/).

```bash
# Create a folder for WYGIWYH (optional)
$ mkdir WYGIWYH

# Go into the folder
$ cd WYGIWYH

$ touch docker-compose.yml
$ nano docker-compose.yml
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs

# Fill the .env file with your configurations
$ touch .env
$ nano .env # or any other editor you want to use
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly

# Run the app
$ docker compose up -d

# Create the first admin account. This isn't required if you set the enviroment variables: ADMIN_EMAIL and ADMIN_PASSWORD.
$ docker compose exec -it web python manage.py createsuperuser
```

>   **Note:** For Unraid users, use the app on the store.  See the [Unraid section](#unraid) and [Environment Variables](#environment-variables) for details.

### Running Locally

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Leave the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

Access the app at `localhost:OUTBOUND_PORT`.

**Important Notes:**

*   Add your machine's IP to `DJANGO_ALLOWED_HOSTS` if running behind services like Tailscale.
*   Include the IP in `DJANGO_ALLOWED_HOSTS` if not using localhost.

### Latest Changes

Features are added to `main` when ready. Use the `:nightly` tag on Docker for the latest version, but be aware of potential breaking changes.

See all required Dockerfiles [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

### Unraid

Thanks to [nwithan8](https://github.com/nwithan8) for providing a Unraid template.  See the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is available on the Unraid Store.  You'll need your own PostgreSQL database (version 15 or up).

Create the first user using the container's console in Unraid's UI: `python manage.py createsuperuser`.

### Environment Variables

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                              |
| :---------------------------- | :---------- | :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | Space-separated list of domains and IPs representing the host/domain names that WYGIWYH can serve.  [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                             |
| HTTPS_ENABLED                 | true\|false | false                             | Whether to use secure cookies. If true, the cookie will be marked as “secure”, which means browsers may ensure that the cookie is only sent under an HTTPS connection                                                                   |
| URL                           | string      | http://localhost http://127.0.0.1 | Space-separated list of domains and IPs (with protocol) representing trusted origins for unsafe requests.  [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                   |
| SECRET_KEY                    | string      | ""                                | Unique, unpredictable value for cryptographic signing.                                                                                                                                                                                    |
| DEBUG                         | true\|false | false                             | Turns DEBUG mode on or off.  Don't use in production.                                                                                                                                                                                    |
| SQL_DATABASE                  | string      | None *required                    | Your PostgreSQL database name.                                                                                                                                                                                                           |
| SQL_USER                      | string      | user                              | Username for connecting to your PostgreSQL database.                                                                                                                                                                                       |
| SQL_PASSWORD                  | string      | password                          | Password for connecting to your PostgreSQL database.                                                                                                                                                                                       |
| SQL_HOST                      | string      | localhost                         | Address for connecting to your PostgreSQL database.                                                                                                                                                                                       |
| SQL_PORT                      | string      | 5432                              | Port for connecting to your PostgreSQL database.                                                                                                                                                                                          |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                          |
| ENABLE_SOFT_DELETE            | true\|false | false                             | Enable soft deletes for transactions.  Deleted transactions remain in the database (useful for imports).                                                                                                                               |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                               | Time in days to keep soft-deleted transactions.  Use with `ENABLE_SOFT_DELETE`.  Set to 0 to keep indefinitely.                                                                                                                           |
| TASK_WORKERS                  | int         | 1                                 | Number of workers for async tasks (one is usually enough).                                                                                                                                                                              |
| DEMO                          | true\|false | false                             | Enable demo mode.                                                                                                                                                                                                                         |
| ADMIN_EMAIL                   | string      | None                              | Automatically create an admin account with this email.  Requires `ADMIN_PASSWORD`.                                                                                                                                                        |
| ADMIN_PASSWORD                | string      | None                              | Automatically create an admin account with this password.  Requires `ADMIN_EMAIL`.                                                                                                                                                        |
| CHECK_FOR_UPDATES             | bool        | true                              | Check for new version updates and notify users. The check is done by doing a single query to Github's API every 12 hours.                                                                                 |

### OIDC Configuration

WYGIWYH supports login via OpenID Connect (OIDC) through `django-allauth`. This allows users to authenticate using an external OIDC provider.

>   **Note:** Currently only OpenID Connect is supported as a provider, open an issue if you need something else.

To configure OIDC, you need to set the following environment variables:

| Variable             | Description                                                                                                                                                                                                                                            |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | The name of the provider.  Will be displayed in the login page.  Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`).  `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

When configuring your OIDC provider, you will need to provide a callback URL (also known as a Redirect URI). For WYGIWYH, the default callback URL is:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with the actual URL where your WYGIWYH instance is accessible. And `<OIDC_CLIENT_NAME>` with the slugfied value set in OIDC_CLIENT_NAME or the default `openid-connect` if you haven't set this variable.

## How it Works

For in-depth information, see the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Translation

Help translate WYGIWYH!

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

>   Login with your GitHub account.

## Caveats and Warnings

*   I'm not an accountant; some terms and calculations might be incorrect. Please open an issue if you find anything that needs improvement.
*   Most calculations are done at runtime, potentially impacting performance.  On my instance (3000+ transactions, 4+ years, 4000+ exchange rates), page load times average ~500ms.
*   This is not a budgeting or double-entry-accounting application.  If you need those features, explore other options.  If you want them in WYGIWYH, open a discussion.

## Built With

WYGIWYH utilizes a range of open-source tools:

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