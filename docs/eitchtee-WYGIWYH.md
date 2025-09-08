<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple, Flexible Approach
  <br>
</h1>

<p align="center">
  <a href="#key-features">Key Features</a> |
  <a href="#why-wygiwyh">Why WYGIWYH?</a> |
  <a href="#getting-started">Getting Started</a> |
  <a href="#usage">Usage</a> |
  <a href="#help-translate">Help Translate</a> |
  <a href="#caveats">Caveats</a>
  <br>
  <a href="https://github.com/eitchtee/WYGIWYH">
  <img alt="GitHub Repo" src="https://img.shields.io/github/stars/eitchtee/WYGIWYH?style=social" />
  </a>
</p>

**WYGIWYH (What You Get Is What You Have)** is a straightforward, opinionated finance tracker designed for those who prefer a no-budget approach.  Manage your money simply and effectively with features like multi-currency support, customizable transactions, and a built-in dollar-cost averaging tracker.

<img src=".github/img/monthly_view.png" width="18%"></img> <img src=".github/img/yearly.png" width="18%"></img> <img src=".github/img/networth.png" width="18%"></img> <img src=".github/img/calendar.png" width="18%"></img> <img src=".github/img/all_transactions.png" width="18%"></img>

## Key Features

*   **Unified Transaction Tracking:**  Centralize all income and expenses in one place.
*   **Multi-Account Support:** Track funds across various accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:**  Seamlessly manage transactions and balances in different currencies, including custom currencies.
*   **Automated Adjustments:** Customize transactions with rules for automation.
*   **Dollar-Cost Averaging (DCA) Tracker:** Track recurring investments for crypto, stocks and more.
*   **API for Automation:** Integrate with other services to sync transactions.

## Why WYGIWYH?

Traditional budgeting can be overly complex. WYGIWYH simplifies money management with a principle-first approach:

> Use what you earn this month for this month.  Savings are tracked, but are considered untouchable for future months.

This philosophy allows you to track your spending without the constraints of budgeting. Built out of frustration with existing tools, WYGIWYH provides the features and flexibility needed to effectively manage finances.

## Getting Started

To run WYGIWYH, you will need [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/)

From your command line:

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

**Note:**  For running locally, remove `URL`, set `HTTPS_ENABLED` to `false`, and keep the default `DJANGO_ALLOWED_HOSTS`.

### Unraid

WYGIWYH is available on the Unraid Store. You'll need to provision your own postgres (version 15 or up) database.

To create the first user, open the container's console using Unraid's UI, by clicking on WYGIWYH icon on the Docker page and selecting `Console`, then type `python manage.py createsuperuser`, you'll them be prompted to input your e-mail and password.

## Usage

Detailed usage instructions can be found in the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

### Demo

You can try WYGIWYH on [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/) with the credentials below:

> [!NOTE]
> E-mail: `demo@demo.com`
> 
> Password: `wygiwyhdemo`

Keep in mind that **any data you add will be wiped in 24 hours or less**. And that **most automation features like the API, Rules, Automatic Exchange Rates and Import/Export are disabled**.

### Environment Variables

The following environment variables can be used to configure the application:

| Variable                      | Type        | Default                           | Explanation                                                                                                                                                                                                                              |
|-------------------------------|-------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | A list of space separated domains and IPs representing the host/domain names that WYGIWYH site can serve. [Click here](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts) for more details                               |
| HTTPS_ENABLED                 | true\|false | false                             | Whether to use secure cookies. If this is set to true, the cookie will be marked as “secure”, which means browsers may ensure that the cookie is only sent under an HTTPS connection                                                     |
| URL                           | string      | http://localhost http://127.0.0.1 | A list of space separated domains and IPs (with the protocol) representing the trusted origins for unsafe requests (e.g. POST). [Click here](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins ) for more details |
| SECRET_KEY                    | string      | ""                                | This is used to provide cryptographic signing, and should be set to a unique, unpredictable value.                                                                                                                                       |
| DEBUG                         | true\|false | false                             | Turns DEBUG mode on or off, this is useful to gather more data about possible errors you're having. Don't use in production.                                                                                                             |
| SQL_DATABASE                  | string      | None *required                    | The name of your postgres database                                                                                                                                                                                                       |
| SQL_USER                      | string      | user                              | The username used to connect to your postgres database                                                                                                                                                                                   |
| SQL_PASSWORD                  | string      | password                          | The password used to connect to your postgres database                                                                                                                                                                                   |
| SQL_HOST                      | string      | localhost                         | The address used to connect to your postgres database                                                                                                                                                                                    |
| SQL_PORT                      | string      | 5432                              | The port used to connect to your postgres database                                                                                                                                                                                       |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                 | The age of session cookies, in seconds. E.g. how long you will stay logged in                                                                                                                                                            |
| ENABLE_SOFT_DELETE            | true\|false | false                             | Whether to enable transactions soft delete, if enabled, deleted transactions will remain in the database. Useful for imports and avoiding duplicate entries.                                                                             |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                               | Time in days to keep soft deleted transactions for. If 0, will keep all transactions indefinitely. Only works if ENABLE_SOFT_DELETE is true.                                                                                             |
| TASK_WORKERS                  | int         | 1                                 | How many workers to have for async tasks. One should be enough for most use cases                                                                                                                                                        |
| DEMO                          | true\|false | false                             | If demo mode is enabled.                                                                                                                                                                                                                 |
| ADMIN_EMAIL                   | string      | None                              | Automatically creates an admin account with this email. Must have `ADMIN_PASSWORD` also set.                                                                                                                                             |
| ADMIN_PASSWORD                | string      | None                              | Automatically creates an admin account with this password. Must have `ADMIN_EMAIL` also set.                                                                                                                                             |
| CHECK_FOR_UPDATES             | bool        | true                              | Check and notify users about new versions. The check is done by doing a single query to Github's API every 12 hours.                                                                                  |

### OIDC Configuration

WYGIWYH supports login via OpenID Connect (OIDC) through `django-allauth`. This allows users to authenticate using an external OIDC provider.

> [!NOTE]
> Currently only OpenID Connect is supported as a provider, open an issue if you need something else.

To configure OIDC, you need to set the following environment variables:

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider. will be displayed in the login page. Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

When configuring your OIDC provider, you will need to provide a callback URL (also known as a Redirect URI). For WYGIWYH, the default callback URL is:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with the actual URL where your WYGIWYH instance is accessible. And `<OIDC_CLIENT_NAME>` with the slugfied value set in OIDC_CLIENT_NAME or the default `openid-connect` if you haven't set this variable.

### Latest Changes

Features are only added to `main` when ready; use the `:nightly` tag on Docker for the latest version. Note that undocumented breaking changes can occur.

## Help Translate

Help translate WYGIWYH via [our translation platform](https://translations.herculino.com/engage/wygiwyh/)!

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

## Caveats

*   The creator is not an accountant; ensure to open an issue if you identify any potential errors.
*   Calculations are performed at runtime, which could impact performance.
*   This app is not a budgeting or double-entry accounting application.

## Built With

WYGIWYH is built with a variety of open-source tools, including: Django, HTMX, _hyperscript, Procrastinate, Bootstrap, Tailwind, Webpack, PostgreSQL, Django REST framework, and Alpine.js.