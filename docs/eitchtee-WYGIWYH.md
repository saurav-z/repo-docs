<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Powerful & Opinionated Finance Tracker
  <br>
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#unraid">Unraid</a> •
  <a href="#environment-variables">Environment Variables</a> •
  <a href="#oidc-configuration">OIDC Configuration</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#help-translate">Help Translate</a> •
  <a href="#caveats">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** (What You Get Is What You Have) offers a straightforward, no-budget approach to personal finance management, helping you understand where your money goes.

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## About

Tired of complex budgeting apps? WYGIWYH simplifies money management by following a simple principle: use what you earn this month for this month. Savings are tracked but treated as untouched for future months. This approach helps you avoid overspending and provides clarity on your financial health.

WYGIWYH was born out of the need for a flexible, multi-currency finance tracker without the constraints of traditional budgeting. The goal? A powerful tool that simplifies, not complicates, personal finance.

## Key Features

*   **Unified Transaction Tracking:** Monitor all income and expenses in one place.
*   **Multi-Account Support:** Track funds across banks, wallets, and investments.
*   **Multi-Currency Support:** Seamlessly manage transactions in various currencies.
*   **Custom Currencies:** Create currencies for crypto, rewards points, or other models.
*   **Automated Adjustments with Rules:** Automate transaction modifications.
*   **Built-in Dollar-Cost Average (DCA) Tracker:** Track recurring investments.
*   **API Support:** Integrate with other services for automation.

## Demo

Explore WYGIWYH's capabilities with our demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

> [!NOTE]
> *   E-mail: `demo@demo.com`
> *   Password: `wygiwyhdemo`
>   
> **Important:** Demo data is reset daily. Most automation features are disabled in the demo.

## Getting Started

To run WYGIWYH, you'll need [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).

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

> [!NOTE]
> If you're using Unraid, you don't need to follow these steps, use the app on the store. Make sure to read the [Unraid section](#unraid) and [Environment Variables](#environment-variables) for an explanation of all available variables

### Running Locally

To run WYGIWYH locally, modify your `.env` file:

1.  Remove `URL`
2.  Set `HTTPS_ENABLED` to `false`
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 \[::1])

Then, access the app via `localhost:OUTBOUND_PORT`.

> [!NOTE]
> *   For use with Tailscale or similar, also add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
> *   For non-localhost IPs, add the IP to `DJANGO_ALLOWED_HOSTS` without `http://`.

### Latest Changes

Features are added to `main` when ready. To use the latest features, build from source or use the `:nightly` Docker tag. Be aware of possible undocumented changes. The required Dockerfiles are [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

## Unraid

Thanks to [nwithan8](https://github.com/nwithan8), a Unraid template is available. See the [unraid\_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is also available on the Unraid Store.  You'll need to provide your own PostgreSQL (version 15 or up) database.  To create the first user, use the container's console (click WYGIWYH icon on the Docker page and select `Console`), then enter `python manage.py createsuperuser`.

## Environment Variables

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                              |
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

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) for login via `django-allauth`.

> [!NOTE]
> Currently only OpenID Connect is supported as a provider, open an issue if you need something else.

Set these environment variables:

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider. will be displayed in the login page. Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

The callback URL (Redirect URI) for WYGIWYH is:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your instance's URL and `<OIDC_CLIENT_NAME>` with the value of the `OIDC_CLIENT_NAME` variable, or `openid-connect` if it is unset.

## How It Works

For more details, see the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Help Translate

Contribute to WYGIWYH's translations:

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your github account

## Caveats and Warnings

*   Not an accountant's tool; some terms and calculations may have errors.  Open an issue if you find anything that needs improvement.
*   Calculations are mostly done at runtime, which might affect performance.
*   Not a budgeting or double-entry accounting application. If you need those, open a discussion.

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