<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a No-Budget Approach
  <br>
</h1>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#demo">Demo</a> •
  <a href="#help-us-translate">Translate</a> •
  <a href="#caveats-and-warnings">Caveats and Warnings</a> •
  <a href="#built-with">Built With</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a powerful, open-source finance tracker designed for users seeking a simple, principles-first approach to money management. [Explore the WYGIWYH repository](https://github.com/eitchtee/WYGIWYH).

<img src=".github/img/monthly_view.png" width="18%">
<img src=".github/img/yearly.png" width="18%">
<img src=".github/img/networth.png" width="18%">
<img src=".github/img/calendar.png" width="18%">
<img src=".github/img/all_transactions.png" width="18%">

## Key Features

WYGIWYH offers a comprehensive suite of features to simplify your financial tracking:

*   **Unified Transaction Tracking:** Effortlessly record all income and expenses in one centralized location.
*   **Multi-Account Support:** Manage your finances across various accounts, including banks, wallets, and investments.
*   **Multi-Currency Support:** Seamlessly handle transactions and balances in different currencies.
*   **Custom Currencies:** Create your own currencies to track crypto, rewards points, or any other model.
*   **Automated Adjustments with Rules:** Automate transaction modifications using customizable rules.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Effectively monitor recurring investments, particularly for crypto and stocks.
*   **API Support for Automation:** Integrate WYGIWYH with other services to automate transaction synchronization.

## Why WYGIWYH?

Tired of complex budgeting apps? WYGIWYH simplifies money management with a straightforward principle:

> Use what you earn this month for this month. Any savings are tracked but treated as untouchable for future months.

This principle minimizes the complexity of tracking your finances while providing a clear overview of your spending and savings.

WYGIWYH was built to address the limitations of existing tools, offering:

*   Multi-currency support
*   A no-budget approach
*   Web application usability
*   An automation-ready API
*   Custom transaction rules

## Demo

Try WYGIWYH before you install!
*   **Email:** `demo@demo.com`
*   **Password:** `wygiwyhdemo`

[Try the WYGIWYH Demo](https://wygiwyh-demo.herculino.com/)

> [!NOTE]
> Data on the demo instance is automatically reset every 24 hours. Most automation features such as the API, Rules, Automatic Exchange Rates, and Import/Export are disabled.

## How To Use

WYGIWYH requires [Docker](https://docs.docker.com/engine/install/) with [docker-compose](https://docs.docker.com/compose/install/) to run.

1.  **Create a directory** (optional): `mkdir WYGIWYH && cd WYGIWYH`
2.  **Create a `docker-compose.yml` file**: `touch docker-compose.yml && nano docker-compose.yml`
    *   Paste the content from [this file](https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml), and edit to match your needs.
3.  **Create an `.env` file**: `touch .env && nano .env`
    *   Populate with configurations from [this example](https://github.com/eitchtee/WYGIWYH/blob/main/.env.example)
4.  **Run the application:** `docker compose up -d`
5.  **Create the first admin user:** `docker compose exec -it web python manage.py createsuperuser`

> [!NOTE]
>  If you are using Unraid, read the Unraid section and Environment Variables sections for instructions on setting it up.

### Running Locally

To run WYGIWYH locally:

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

You can then access the application via `localhost:OUTBOUND_PORT`.

> [!NOTE]
> If running behind a service such as Tailscale, add the machine's IP address to `DJANGO_ALLOWED_HOSTS`. If using a different IP, include it in `DJANGO_ALLOWED_HOSTS`.

### Latest Changes
To run the latest version, build from source or use the `:nightly` tag on Docker. Be aware of potential breaking changes.

All required Dockerfiles can be found [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

## Unraid

WYGIWYH has a dedicated Unraid template available. You can find it in the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is available on the Unraid Store. You'll need to provision your own Postgres (version 15 or up) database.

To create the first user, open the container's console using Unraid's UI, by clicking on WYGIWYH icon on the Docker page and selecting `Console`, then type `python manage.py createsuperuser`, you'll them be prompted to input your e-mail and password.

## Environment Variables

Configure WYGIWYH using the following environment variables:

| Variable                       | Type        | Default                           | Description                                                                                                                                                                                                                                  |
| ------------------------------ | ----------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DJANGO_ALLOWED_HOSTS`         | string      | `localhost 127.0.0.1`               | A space-separated list of domains and IPs for the allowed hosts. [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                                                                              |
| `HTTPS_ENABLED`                | true\|false | `false`                           | Enables secure cookies.  If set to `true`, cookies will only be sent over HTTPS.                                                                                                                                                            |
| `URL`                          | string      | `http://localhost http://127.0.0.1` | A space-separated list of domains and IPs (with the protocol) representing the trusted origins for unsafe requests. [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                           |
| `SECRET_KEY`                   | string      | `""`                              | Cryptographic signing key; should be unique and unpredictable.                                                                                                                                                                                 |
| `DEBUG`                        | true\|false | `false`                           | Enables/disables debug mode. Use for troubleshooting; do not use in production.                                                                                                                                                              |
| `SQL_DATABASE`                 | string      | None *required                    | The name of your Postgres database.                                                                                                                                                                                                           |
| `SQL_USER`                     | string      | `user`                            | Your Postgres database username.                                                                                                                                                                                                              |
| `SQL_PASSWORD`                 | string      | `password`                        | Your Postgres database password.                                                                                                                                                                                                              |
| `SQL_HOST`                     | string      | `localhost`                       | The address of your Postgres database host.                                                                                                                                                                                                   |
| `SQL_PORT`                     | string      | `5432`                            | The port for your Postgres database.                                                                                                                                                                                                          |
| `SESSION_EXPIRY_TIME`          | int         | `2678400` (31 days)                 | Session cookie expiry time in seconds.                                                                                                                                                                                                      |
| `ENABLE_SOFT_DELETE`           | true\|false | `false`                           | Enables soft-deleting transactions (deleted transactions remain in the database). Useful for imports and avoiding duplicates.                                                                                                              |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | `365`                             | Time in days to keep soft-deleted transactions.  `0` means indefinitely.  Only active if `ENABLE_SOFT_DELETE` is `true`.                                                                                                                  |
| `TASK_WORKERS`                 | int         | `1`                                 | Number of workers for asynchronous tasks.                                                                                                                                                                                                     |
| `DEMO`                         | true\|false | `false`                           | Enables demo mode.                                                                                                                                                                                                                            |
| `ADMIN_EMAIL`                  | string      | None                              | Automatically creates an admin account with this email. Requires `ADMIN_PASSWORD`.                                                                                                                                                         |
| `ADMIN_PASSWORD`               | string      | None                              | Automatically creates an admin account with this password. Requires `ADMIN_EMAIL`.                                                                                                                                                         |
| `CHECK_FOR_UPDATES`            | bool        | `true`                            | Checks for and notifies users about new versions. Checks Github's API every 12 hours.                                                                                                                                                     |

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) login via `django-allauth`.

> [!NOTE]
> Currently, only OpenID Connect is supported.

Set these environment variables to configure OIDC:

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider.  Shown in the login page. Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | Your OIDC provider's discovery document URL (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` uses this to find endpoints. |
| `OIDC_ALLOW_SIGNUP`  | Allows automatic creation of accounts after a successful authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

Configure your OIDC provider with the following callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your actual WYGIWYH instance URL. And `<OIDC_CLIENT_NAME>` with the slugfied value set in OIDC_CLIENT_NAME or the default `openid-connect` if you haven't set this variable.

## How it Works

Learn more about WYGIWYH's inner workings in the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Help Us Translate WYGIWYH!

Contribute to WYGIWYH's global reach by helping translate the application.

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your Github account

## Caveats and Warnings

*   I am not a professional accountant; calculations and terminology might have errors. Open an issue if you notice any improvements.
*   Most calculations are performed at runtime, which can impact performance. While I have experience with 3,000+ transactions and 4,000+ exchange rates, and my average load times are around 500ms per page, it is something to note.
*   WYGIWYH is not a budgeting or double-entry accounting application. If you require those features, explore other options.

## Built With

WYGIWYH is built using the following open-source tools:

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