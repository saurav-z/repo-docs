<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Finance Tracker for Simple, Flexible Money Management
  <br>
</h1>

<h4 align="center">Take control of your finances with WYGIWYH, a powerful, opinionated finance tracker.</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#demo">Demo</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#help-us-translate">Help Translate</a> •
  <a href="#caveats-and-warnings">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a finance tracker designed for those who prefer a straightforward, no-budget approach to managing their finances. WYGIWYH simplifies financial tracking with features like multi-currency support, custom transaction rules, and a built-in dollar-cost averaging tracker.

<img src=".github/img/monthly_view.png" width="18%"></img> <img src=".github/img/yearly.png" width="18%"></img> <img src=".github/img/networth.png" width="18%"></img> <img src=".github/img/calendar.png" width="18%"></img> <img src=".github/img/all_transactions.png" width="18%"></img> 

## Key Features

*   **Unified Transaction Tracking:** Record all income and expenses in one place.
*   **Multi-Account Support:** Track money and assets across banks, wallets, and investments.
*   **Out-of-the-Box Multi-Currency Support:** Manage transactions and balances in different currencies.
*   **Custom Currencies:** Create currencies for crypto, rewards points, and more.
*   **Automated Adjustments with Rules:** Automatically modify transactions with customizable rules.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Track recurring investments.
*   **API Support for Automation:** Integrate with existing services to synchronize transactions.

## Why WYGIWYH?

WYGIWYH (pronounced "wiggy-wih") simplifies money management with a core principle:

> Use what you earn this month for this month. Savings are tracked but treated as untouchable for future months.

This philosophy avoids budgeting constraints while tracking your spending.  WYGIWYH was born out of the need for a financial tool that offered multi-currency support, web app usability, automation capabilities, and custom transaction rules.

## Demo

Explore WYGIWYH's features with a live demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

> [!NOTE]
> E-mail: `demo@demo.com`
> 
> Password: `wygiwyhdemo`

*   Any data added in the demo will be wiped in 24 hours or less.
*   Most automation features are disabled.

## How To Use

WYGIWYH requires [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).

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

### Running Locally

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep default `DJANGO_ALLOWED_HOSTS`.
4.  Access via `localhost:OUTBOUND_PORT`.

> [!NOTE]
> - If running behind Tailscale or similar, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
> - Add non-localhost IPs to `DJANGO_ALLOWED_HOSTS`.

### Latest Changes

Features are added to `main` when ready. Use the `:nightly` Docker tag for the latest. Be aware of potential breaking changes.

All the required Dockerfiles are [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

## Unraid

[nwithan8](https://github.com/nwithan8) provides a Unraid template; see the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is also on the Unraid Store. You'll need your own PostgreSQL database (version 15+).

Create the first user using the Unraid UI console (Docker page > WYGIWYH icon > `Console`): `python manage.py createsuperuser`.

## Environment Variables

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                              |
| :---------------------------- | :---------- | :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | Comma-separated domains/IPs the WYGIWYH site can serve.  [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                                                                                    |
| HTTPS_ENABLED                 | true\|false | false                             | Use secure cookies. If true, cookies are marked "secure," sent only over HTTPS.                                                                                                                                                             |
| URL                           | string      | http://localhost http://127.0.0.1 | Comma-separated domains/IPs (with protocol) for trusted origins for unsafe requests (e.g., POST). [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                                       |
| SECRET_KEY                    | string      | ""                                | Unique, unpredictable cryptographic signing key.                                                                                                                                                                                             |
| DEBUG                         | true\|false | false                             | Enable or disable DEBUG mode. Don't use in production.                                                                                                                                                                                         |
| SQL_DATABASE                  | string      | None *required                    | PostgreSQL database name.                                                                                                                                                                                                                    |
| SQL_USER                      | string      | user                              | PostgreSQL username.                                                                                                                                                                                                                         |
| SQL_PASSWORD                  | string      | password                          | PostgreSQL password.                                                                                                                                                                                                                         |
| SQL_HOST                      | string      | localhost                         | PostgreSQL host address.                                                                                                                                                                                                                     |
| SQL_PORT                      | string      | 5432                              | PostgreSQL port.                                                                                                                                                                                                                             |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                 | Session cookie age in seconds (e.g., login duration).                                                                                                                                                                                        |
| ENABLE_SOFT_DELETE            | true\|false | false                             | Enable soft deletes for transactions (deleted transactions remain in the database). Useful for imports/duplicate avoidance.                                                                                                                   |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                               | Days to keep soft-deleted transactions. 0 = keep indefinitely. Only works if `ENABLE_SOFT_DELETE` is true.                                                                                                                                   |
| TASK_WORKERS                  | int         | 1                                 | Number of workers for async tasks.                                                                                                                                                                                                           |
| DEMO                          | true\|false | false                             | Enable demo mode.                                                                                                                                                                                                                            |
| ADMIN_EMAIL                   | string      | None                              | Automatically creates an admin account with this email (requires `ADMIN_PASSWORD`).                                                                                                                                                          |
| ADMIN_PASSWORD                | string      | None                              | Automatically creates an admin account with this password (requires `ADMIN_EMAIL`).                                                                                                                                                         |
| CHECK_FOR_UPDATES             | bool        | true                              | Check for and notify users about new versions.  A single query to Github's API is done every 12 hours.                                                                                                                                         |

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) via `django-allauth`.

| Variable             | Description                                                                                                                                                                                                                                            |
| :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | Provider name displayed in the login page. Defaults to `OpenID Connect`.                                                                                                                                                                               |
| `OIDC_CLIENT_ID`     | Client ID from your OIDC provider.                                                                                                                                                                                                                       |
| `OIDC_CLIENT_SECRET` | Client Secret from your OIDC provider.                                                                                                                                                                                                                   |
| `OIDC_SERVER_URL`    | Base URL of your OIDC provider's discovery document or authorization server. Used by `django-allauth` to discover endpoints.                                                                                                                          |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic creation of accounts upon successful authentication. Defaults to `true`.                                                                                                                                                               |

**Callback URL:**

Configure your OIDC provider with this callback URL (Redirect URI):

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` and `<OIDC_CLIENT_NAME>` with the appropriate values.

## How it Works

See the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki) for more.

## Help Us Translate

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your github account

## Caveats and Warnings

-   I'm not an accountant; some terms/calculations may be inaccurate. Please report issues.
-   Most calculations are done at runtime, potentially impacting performance. Load times are around 500ms on a personal instance with many transactions/exchange rates.
-   This is not a budgeting or double-entry accounting app.  Suggest desired features via discussion.

## Built with

WYGIWYH utilizes these open-source tools:

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