<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker
  <br>
</h1>

<p align="center">
  <a href="#key-features">Key Features</a> |
  <a href="#why-wygiwyh">Why WYGIWYH?</a> |
  <a href="#how-to-use">How to Use</a> |
  <a href="#demo">Demo</a> |
  <a href="#help-us-translate">Translate</a> |
  <a href="#contributing">Contributing</a>
</p>

**WYGIWYH (What You Get Is What You Have)** is a finance tracker designed for users who prefer a straightforward, no-budget approach to managing their money. [**Explore WYGIWYH on GitHub**](https://github.com/eitchtee/WYGIWYH)

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## Key Features

*   **Unified Transaction Tracking:**  Record all income and expenses in one place.
*   **Multi-Account Support:** Track money across banks, wallets, and investments.
*   **Multi-Currency Support:** Manage transactions and balances in different currencies.
*   **Custom Currencies:** Create your own currencies for crypto, rewards points, or more.
*   **Automated Adjustments with Rules:** Automatically modify transactions with customizable rules.
*   **Built-in Dollar-Cost Average (DCA) Tracker:** Easily track recurring investments.
*   **API Support:** Integrate with other services for automation.

## Why WYGIWYH?

WYGIWYH is built on a simple principle: **Use what you earn this month for this month.**  It's designed for users who want to avoid complex budgeting and focus on tracking where their money goes. The goal is to provide a clean and flexible financial tracking experience.

## Demo

Experience WYGIWYH firsthand at [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/) with the following credentials:

> [!NOTE]
> E-mail: `demo@demo.com`
> 
> Password: `wygiwyhdemo`

**Note:** Data added to the demo is wiped every 24 hours or less, and many automation features are disabled.

## How to Use

WYGIWYH is a powerful self-hosted application. It requires Docker and docker-compose.

### Getting Started:

```bash
# Create a directory (optional)
mkdir WYGIWYH
cd WYGIWYH

# Create docker-compose.yml
touch docker-compose.yml
nano docker-compose.yml
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml
# and edit according to your needs

# Create .env file
touch .env
nano .env # or any other editor you want to use
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example
# and edit accordingly

# Run the application
docker compose up -d

# Create an admin account (if not using ADMIN_EMAIL and ADMIN_PASSWORD env vars)
docker compose exec -it web python manage.py createsuperuser
```

### Running Locally

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

Access WYGIWYH locally at `localhost:OUTBOUND_PORT`.

> [!NOTE]
> -   If running behind Tailscale or similar, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
> -   If using an IP other than localhost, add it to `DJANGO_ALLOWED_HOSTS` without `http://`.

### Latest Changes

For the latest features, build from source or use the `:nightly` tag on Docker.  Check the [Dockerfiles](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod) for details.

### Unraid

WYGIWYH is available on the Unraid Store. You'll need your own Postgres database (version 15 or up).  See the [Unraid section](#unraid) and [Environment Variables](#environment-variables) for configuration details.  The Unraid Docker template can be found in the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

### Environment Variables

A comprehensive list of available environment variables with explanations can be found below.

| Variable                   | Type        | Default                           | Explanation                                                                                                                                                                                                                              |
| :------------------------- | :---------- | :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DJANGO\_ALLOWED\_HOSTS   | string      | localhost 127.0.0.1               | A list of space separated domains and IPs representing the host/domain names that WYGIWYH site can serve. [Click here](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts) for more details                               |
| HTTPS\_ENABLED             | true\|false | false                             | Whether to use secure cookies. If this is set to true, the cookie will be marked as “secure”, which means browsers may ensure that the cookie is only sent under an HTTPS connection                                                     |
| URL                        | string      | http://localhost http://127.0.0.1 | A list of space separated domains and IPs (with the protocol) representing the trusted origins for unsafe requests (e.g. POST). [Click here](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins ) for more details |
| SECRET\_KEY                | string      | ""                                | This is used to provide cryptographic signing, and should be set to a unique, unpredictable value.                                                                                                                                       |
| DEBUG                      | true\|false | false                             | Turns DEBUG mode on or off, this is useful to gather more data about possible errors you're having. Don't use in production.                                                                                                             |
| SQL\_DATABASE              | string      | None *required                    | The name of your postgres database                                                                                                                                                                                                       |
| SQL\_USER                  | string      | user                              | The username used to connect to your postgres database                                                                                                                                                                                   |
| SQL\_PASSWORD              | string      | password                          | The password used to connect to your postgres database                                                                                                                                                                                   |
| SQL\_HOST                  | string      | localhost                         | The address used to connect to your postgres database                                                                                                                                                                                    |
| SQL\_PORT                  | string      | 5432                              | The port used to connect to your postgres database                                                                                                                                                                                       |
| SESSION\_EXPIRY\_TIME      | int         | 2678400 (31 days)                 | The age of session cookies, in seconds. E.g. how long you will stay logged in                                                                                                                                                            |
| ENABLE\_SOFT\_DELETE       | true\|false | false                             | Whether to enable transactions soft delete, if enabled, deleted transactions will remain in the database. Useful for imports and avoiding duplicate entries.                                                                             |
| KEEP\_DELETED\_TRANSACTIONS\_FOR | int | 365 | Time in days to keep soft deleted transactions for. If 0, will keep all transactions indefinitely. Only works if ENABLE\_SOFT\_DELETE is true. |
| TASK\_WORKERS              | int         | 1                                 | How many workers to have for async tasks. One should be enough for most use cases                                                                                                                                                        |
| DEMO                       | true\|false | false                             | If demo mode is enabled.                                                                                                                                                                                                                 |
| ADMIN\_EMAIL               | string      | None                              | Automatically creates an admin account with this email. Must have `ADMIN_PASSWORD` also set.                                                                                                                                             |
| ADMIN\_PASSWORD            | string      | None                              | Automatically creates an admin account with this password. Must have `ADMIN_EMAIL` also set.                                                                                                                                             |
| CHECK\_FOR\_UPDATES        | bool        | true                              | Check and notify users about new versions. The check is done by doing a single query to Github's API every 12 hours.                                                                                                                 |

### OIDC Configuration

WYGIWYH supports login via OpenID Connect (OIDC) through `django-allauth`. This allows users to authenticate using an external OIDC provider.

> [!NOTE]
> Currently only OpenID Connect is supported as a provider, open an issue if you need something else.

To configure OIDC, you need to set the following environment variables:

| Variable             | Description                                                                                                                                                                                                                                            |
| :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | The name of the provider. will be displayed in the login page. Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

When configuring your OIDC provider, you will need to provide a callback URL (also known as a Redirect URI). For WYGIWYH, the default callback URL is:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with the actual URL where your WYGIWYH instance is accessible. And `<OIDC_CLIENT_NAME>` with the slugfied value set in OIDC_CLIENT_NAME or the default `openid-connect` if you haven't set this variable.

## How it Works

For more detailed information, explore the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Help Us Translate

Help translate WYGIWYH:

<a href="https://translations.herculino.com/engage/wygiwyh/">
  <img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your GitHub account.

## Caveats and Warnings

*   I'm not an accountant, some terms and even calculations might be wrong. Open an issue if you see something that can be improved.
*   Most calculations are done at runtime, which can impact performance.  (Load times average around 500ms on the developer's instance.)
*   This is not a budgeting or double-entry-accounting application. If you need those features, please open a discussion.

## Built With

WYGIWYH is built using a combination of powerful open-source tools:

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