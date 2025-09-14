<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Simple & Powerful Finance Tracker
  <br>
</h1>

<p align="center">
  <a href="#key-features">Key Features</a> |
  <a href="#why-wygiwyh">Why WYGIWYH?</a> |
  <a href="#how-to-use">How to Use</a> |
  <a href="#demo">Demo</a> |
  <a href="#contributing">Contribute</a>
</p>

**Tired of complex budgeting apps? WYGIWYH (What You Get Is What You Have) offers a straightforward, no-budget finance tracking experience, allowing you to easily manage your money.**

<img src=".github/img/monthly_view.png" width="18%"></img> <img src=".github/img/yearly.png" width="18%"></img> <img src=".github/img/networth.png" width="18%"></img> <img src=".github/img/calendar.png" width="18%"></img> <img src=".github/img/all_transactions.png" width="18%"></img>

## Key Features

*   **Unified Transaction Tracking:** Easily record all income and expenses in one place.
*   **Multi-Account Support:** Track funds across banks, wallets, and investments.
*   **Multi-Currency Support:** Manage transactions and balances in multiple currencies.
*   **Custom Currencies:** Define your own currencies for crypto, rewards, and more.
*   **Automated Transaction Rules:** Automatically adjust transactions with customizable rules.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Simplify tracking of recurring investments.
*   **API Support:** Integrate with other services for automation.

## Why WYGIWYH?

WYGIWYH is built on a simple principle: **Use what you earn this month for this month.** This helps you avoid overspending while providing insights into your finances.

Traditional budgeting apps can be complex. WYGIWYH solves this by offering a simplified approach with the features you need:

1.  **Multi-currency support**
2.  **No budgeting constraints**
3.  **Web app usability with mobile support (optional)**
4.  **Automation-ready API**
5.  **Custom transaction rules**

## Demo

Explore WYGIWYH with our demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

> [!NOTE]
> E-mail: `demo@demo.com`
>
> Password: `wygiwyhdemo`

Keep in mind that **any data you add will be wiped in 24 hours or less**. And that **most automation features like the API, Rules, Automatic Exchange Rates and Import/Export are disabled**.

## How to Use

To get started, you'll need [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).

1.  **Create a Project Directory:** `mkdir WYGIWYH && cd WYGIWYH`
2.  **Create `docker-compose.yml`:**  `touch docker-compose.yml && nano docker-compose.yml` (Paste the contents of [docker-compose.prod.yml](https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml), and edit for your needs).
3.  **Create `.env`:**  `touch .env && nano .env` (Paste the contents of [.env.example](https://github.com/eitchtee/WYGIWYH/blob/main/.env.example), and edit accordingly).
4.  **Run the App:** `docker compose up -d`
5.  **Create Admin Account (Optional):** `docker compose exec -it web python manage.py createsuperuser`

   *Alternatively, set the `ADMIN_EMAIL` and `ADMIN_PASSWORD` environment variables.*

### Running Locally

To run WYGIWYH locally:

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

You can then access the app at `localhost:OUTBOUND_PORT`.

> [!NOTE]
> *   If you're using a service like Tailscale, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
> *   For non-localhost IPs, add them to `DJANGO_ALLOWED_HOSTS` without `http://`.

### Latest Changes

For the latest features, build from source or use the `:nightly` tag. Be aware of potential breaking changes. Dockerfiles are [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

### Unraid

WYGIWYH is available on the Unraid Store.  You'll need to provision your own PostgreSQL database (version 15 or up).

For more information, please refer to [Unraid Section](#unraid) and [Environment Variables](#environment-variables).

### Environment Variables

| Variable                      | Type        | Default                            | Description                                                                                                                                                                                                                                                  |
| ----------------------------- | ----------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1                | Space-separated domains and IPs for trusted hosts. [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                                                                                                             |
| HTTPS_ENABLED                 | true\|false | false                              | Enable secure cookies.  When set to true, the cookie will be marked as “secure”, which means browsers may ensure that the cookie is only sent under an HTTPS connection                                                                                                                            |
| URL                           | string      | http://localhost http://127.0.0.1    | Space-separated trusted origins for unsafe requests (e.g., POST).  [More info](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                                                                                                   |
| SECRET_KEY                    | string      | ""                                 |  A unique, unpredictable value used for cryptographic signing.                                                                                                                                                                                              |
| DEBUG                         | true\|false | false                              | Enable debug mode (do not use in production).                                                                                                                                                                                                              |
| SQL_DATABASE                  | string      | None *required                     |  PostgreSQL database name.                                                                                                                                                                                                                                   |
| SQL_USER                      | string      | user                               | PostgreSQL username.                                                                                                                                                                                                                                          |
| SQL_PASSWORD                  | string      | password                           | PostgreSQL password.                                                                                                                                                                                                                                          |
| SQL_HOST                      | string      | localhost                          | PostgreSQL host address.                                                                                                                                                                                                                                      |
| SQL_PORT                      | string      | 5432                               | PostgreSQL port.                                                                                                                                                                                                                                            |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                  | Session cookie age in seconds.                                                                                                                                                                                                                             |
| ENABLE_SOFT_DELETE            | true\|false | false                              | Enable soft deletion for transactions.                                                                                                                                                                                                                        |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                                | Days to keep soft-deleted transactions. 0 for indefinite. Only works if `ENABLE_SOFT_DELETE` is true.                                                                                                                                                       |
| TASK_WORKERS                  | int         | 1                                  | Number of workers for async tasks.                                                                                                                                                                                                                           |
| DEMO                          | true\|false | false                              | Enable demo mode.                                                                                                                                                                                                                                            |
| ADMIN_EMAIL                   | string      | None                               | Automatically creates an admin account with this email if `ADMIN_PASSWORD` is also set.                                                                                                                                                                 |
| ADMIN_PASSWORD                | string      | None                               | Automatically creates an admin account with this password if `ADMIN_EMAIL` is also set.                                                                                                                                                                   |
| CHECK_FOR_UPDATES             | bool        | true                               | Check for and notify about new versions.  The check is done by doing a single query to Github's API every 12 hours.                                                                                                                                          |

### OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) login using `django-allauth`.

| Variable             | Description                                                                                                                                                                                                                                               |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider.  Will be displayed in the login page. Defaults to `OpenID Connect`                                                                                                                                                              |
| `OIDC_CLIENT_ID`     | Your OIDC provider's Client ID.                                                                                                                                                                                                                           |
| `OIDC_CLIENT_SECRET` | Your OIDC provider's Client Secret.                                                                                                                                                                                                                       |
| `OIDC_SERVER_URL`    | OIDC provider's base URL (discovery document/authorization server).  (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                                  |

**Callback URL (Redirect URI):**

Configure your OIDC provider with the following callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` and `<OIDC_CLIENT_NAME>` (defaults to `openid-connect` if not set).

## How it Works

For more details, visit our [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Contributing

Help translate WYGIWYH!  [Click here](https://translations.herculino.com/engage/wygiwyh/) to contribute.

> [!NOTE]
> Login with your github account

## Caveats and Warnings

*   I'm not an accountant; some terms and calculations may be inaccurate. Please open an issue if you find anything that needs improvement.
*   Calculations are primarily done at runtime, which can impact performance.
*   This is not a budgeting or double-entry accounting application.

## Built With

WYGIWYH is built with the help of several open-source tools, including:

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

**[View the GitHub Repository](https://github.com/eitchtee/WYGIWYH)**