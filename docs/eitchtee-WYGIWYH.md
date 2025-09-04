<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Simplified Finance Tracker
  <br>
</h1>

<h4 align="center">Take control of your finances with WYGIWYH, a powerful, no-budget finance tracker.</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#demo">Demo</a> •
  <a href="#translations">Translations</a> •
  <a href="#caveats-and-warnings">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a finance tracker designed for users who prefer a straightforward, no-budget approach to managing their money. WYGIWYH simplifies financial management with features like multi-currency support, customizable transactions, and a built-in dollar-cost averaging (DCA) tracker.  [Explore the project on GitHub](https://github.com/eitchtee/WYGIWYH).

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%"> 

## Key Features

*   **Unified Transaction Tracking:**  Record all income and expenses in one place.
*   **Multi-Account Support:** Track funds across banks, wallets, and investments.
*   **Multi-Currency Support:** Manage transactions and balances in different currencies.
*   **Custom Currencies:** Create custom currencies for crypto, rewards, and more.
*   **Automated Adjustments with Rules:** Automate transaction modifications with customizable rules.
*   **Built-in Dollar-Cost Average (DCA) Tracker:** Easily track recurring investments.
*   **API Support:** Integrate with other services for automated transaction synchronization.

## Why WYGIWYH?

WYGIWYH simplifies money management by adhering to a simple principle:

> Use what you earn this month for this month.  Savings are tracked but treated as untouchable for future months.

This philosophy helps you avoid dipping into savings while providing clarity on spending.  WYGIWYH was created to address the lack of financial tools that met the following needs:

1.  Multi-currency support
2.  Non-budgeting approach
3.  Web app usability with mobile support
4.  API for integration
5.  Custom transaction rules

## How to Use

WYGIWYH is built with Docker and Docker Compose.

**Prerequisites:**

*   [Docker](https://docs.docker.com/engine/install/)
*   [docker-compose](https://docs.docker.com/compose/install/)

**Installation:**

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

1.  In your `.env` file:
    *   Remove `URL`
    *   Set `HTTPS_ENABLED` to `false`
    *   Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1])
2.  Access the application at `localhost:OUTBOUND_PORT`.

> [!NOTE]
> - If you're planning on running this behind Tailscale or other similar service also add your machine given IP to `DJANGO_ALLOWED_HOSTS`
> - If you're going to use another IP that isn't localhost, add it to `DJANGO_ALLOWED_HOSTS`, without `http://`

### Latest Changes

Features are only added to `main` when ready, if you want to run the latest version, you must build from source or use the `:nightly` tag on docker. Keep in mind that there can be undocumented breaking changes.

All the required Dockerfiles are [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

### Unraid

[nwithan8](https://github.com/nwithan8) has kindly provided a Unraid template for WYGIWYH, have a look at the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is available on the Unraid Store. You'll need to provision your own postgres (version 15 or up) database.

To create the first user, open the container's console using Unraid's UI, by clicking on WYGIWYH icon on the Docker page and selecting `Console`, then type `python manage.py createsuperuser`, you'll them be prompted to input your e-mail and password.

### Environment Variables

| Variable                      | Type        | Default                           | Explanation                                                                                                                                                                                                                              |
|-------------------------------|-------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | A list of space separated domains and IPs representing the host/domain names that WYGIWYH site can serve. [Click here](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts) for more details                               |
| HTTPS_ENABLED                 | true\|false | false                             | Whether to use secure cookies. If this is set to true, the cookie will be marked as “secure”, which means browsers may ensure that the cookie is only sent under an HTTPS connection                                                     |
| URL                           | string      | http://localhost http://127.0.0.1 | A list of space separated domains and IPs (with the protocol) representing the trusted origins for unsafe requests (e.g. POST). [Click here](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins ) for more details |
| SECRET_KEY                    | string      | ""                                | This is used to provide cryptographic signing, and should be set to a unique, unpredictable value.                                                                                                                                       |
| DEBUG                         | true\|false | false                             | Turns DEBUG mode on or off, this is useful to gather more data about possible errors you're having. Don't use in production.                                                                                                             |
| SQL_DATABASE                  | string      | None *required                    | The name of your postgres database                                                                                                                                                                                                       |
| SQL_USER                      | string      | user                              | The username used to connect to your postgres database                                                                                                                                                                                   |
| SQL_PASSWORD                | string      | password                          | The password used to connect to your postgres database                                                                                                                                                                                   |
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

## Demo

Experience WYGIWYH firsthand!  Try the demo at [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/) with the credentials below:

> [!NOTE]
> E-mail: `demo@demo.com`
> 
> Password: `wygiwyhdemo`

*   **Important:** Demo data is reset every 24 hours.  API, rules, automatic exchange rates, and import/export features are disabled in the demo.

## Translations

Help make WYGIWYH available to more people!  Contribute to translations at:  <a href="https://translations.herculino.com/engage/wygiwyh/"> <img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" /> </a>

> [!NOTE]
> Login with your github account

## Caveats and Warnings

*   I am not an accountant, terms and calculations may be inaccurate.  Report any issues.
*   Calculations are primarily done at runtime, potentially impacting performance.
*   This is not a budgeting or double-entry accounting application.

## Built with

WYGIWYH is built using these open-source technologies:

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