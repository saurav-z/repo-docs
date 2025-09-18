<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Finance Tracker
  <br>
</h1>

<h4 align="center">Take control of your finances with a simple, powerful, and opinionated tracker.</h4>

<p align="center">
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#demo">Demo</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#translate">Translate</a> •
  <a href="#caveats-and-warnings">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH (What You Get Is What You Have)** is a straightforward and flexible finance tracker designed for those who prefer a no-budget approach to money management.  It simplifies finance tracking with features like multi-currency support, customizable transactions, and a built-in dollar-cost averaging tracker.

<img src=".github/img/monthly_view.png" width="18%"></img> <img src=".github/img/yearly.png" width="18%"></img> <img src=".github/img/networth.png" width="18%"></img> <img src=".github/img/calendar.png" width="18%"></img> <img src=".github/img/all_transactions.png" width="18%"></img>

## Why WYGIWYH?

Frustrated with complex budgeting apps, WYGIWYH focuses on a simple principle: "Use what you earn this month for this month." This helps you manage your finances without the constraints of traditional budgeting.

WYGIWYH was created to solve the following needs:

1.  **Multi-currency support** for tracking income and expenses in various currencies.
2.  **No budgeting** - a focus on simplicity and tracking, not strict budgeting.
3.  **Web app usability** with optional mobile support.
4.  **API for automation** to integrate with other tools.
5.  **Customizable transaction rules** to handle various financial scenarios.

## Key Features

WYGIWYH offers an array of features designed to simplify and streamline your personal finance tracking:

*   **Unified Transaction Tracking:**  Record all income and expenses in one place.
*   **Multiple Account Support:**  Track money and assets across different accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:** Manage transactions and balances in different currencies.
*   **Custom Currencies:** Create your own currencies for crypto, rewards points, or any other models.
*   **Automated Rules:**  Automatically modify transactions with customizable rules.
*   **Dollar-Cost Averaging (DCA) Tracker:**  Track recurring investments, especially for crypto and stocks.
*   **API Support:**  Integrate with other services to automate transaction synchronization.

## Demo

Try out WYGIWYH with a live demo.

[wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)
*   **Email:** `demo@demo.com`
*   **Password:** `wygiwyhdemo`

**Important:**  Demo data is wiped every 24 hours or less. Automation features are disabled.

## How To Use

WYGIWYH uses Docker and Docker Compose.

1.  **Set up Docker and Docker Compose:**  Ensure you have [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed.
2.  **Create a Project Directory (Optional):**

    ```bash
    mkdir WYGIWYH
    cd WYGIWYH
    ```
3.  **Create docker-compose.yml:**

    ```bash
    touch docker-compose.yml
    nano docker-compose.yml
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs
    ```
4.  **Configure .env:**

    ```bash
    touch .env
    nano .env
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly
    ```
5.  **Run the application:**

    ```bash
    docker compose up -d
    ```

6.  **Create First Admin User (if environment variables for admin are not set):**

    ```bash
    docker compose exec -it web python manage.py createsuperuser
    ```

### Running Locally

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Leave the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

You can now access `localhost:OUTBOUND_PORT`.

>   **Note:**  If running behind a service like Tailscale, add your machine's IP to `DJANGO_ALLOWED_HOSTS`. Also, add the IP to `DJANGO_ALLOWED_HOSTS` if using an IP that isn't localhost.

### Latest Changes

Features are added to the `main` branch when ready.  Use the `:nightly` tag for the latest version, but be aware of potential breaking changes.  All Dockerfiles are available [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

## Unraid

WYGIWYH is available as an app in the Unraid store. You will need to provide your own PostgreSQL (version 15 or higher) database.

To create the first user, go to the Unraid Docker UI, select the container and open the console, then enter: `python manage.py createsuperuser`.

### Environment Variables

| Variable                      | Type        | Default                                  | Description                                                                                                                                                                                                                       |
|-------------------------------|-------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1                      | A list of domains/IPs the site can serve. [Learn More](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                                                                                      |
| HTTPS_ENABLED                 | true\|false | false                                    | Enable secure cookies (HTTPS).                                                                                                                                                                                                    |
| URL                           | string      | http://localhost http://127.0.0.1          | Trusted origins for unsafe requests (e.g., POST). [Learn More](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                                                                        |
| SECRET_KEY                    | string      | ""                                       | Cryptographic signing key. Must be unique and unpredictable.                                                                                                                                                                     |
| DEBUG                         | true\|false | false                                    | Enable/disable debug mode (for development). **Do not use in production.**                                                                                                                                                         |
| SQL_DATABASE                  | string      | None *required                         | The name of your PostgreSQL database.                                                                                                                                                                                             |
| SQL_USER                      | string      | user                                     | PostgreSQL username.                                                                                                                                                                                                              |
| SQL_PASSWORD                  | string      | password                                 | PostgreSQL password.                                                                                                                                                                                                              |
| SQL_HOST                      | string      | localhost                                | PostgreSQL host address.                                                                                                                                                                                                          |
| SQL_PORT                      | string      | 5432                                     | PostgreSQL port.                                                                                                                                                                                                                |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                        | Session cookie expiration time in seconds.                                                                                                                                                                                        |
| ENABLE_SOFT_DELETE            | true\|false | false                                    | Enable soft deletion for transactions.                                                                                                                                                                                            |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                                      | Days to keep soft-deleted transactions.  If 0, keep indefinitely.  Requires `ENABLE_SOFT_DELETE`.                                                                                                                                |
| TASK_WORKERS                  | int         | 1                                        | Number of workers for async tasks.                                                                                                                                                                                                |
| DEMO                          | true\|false | false                                    | Enable demo mode.                                                                                                                                                                                                                   |
| ADMIN_EMAIL                   | string      | None                                     | Automatically create an admin account with this email. Requires `ADMIN_PASSWORD`.                                                                                                                                                   |
| ADMIN_PASSWORD                | string      | None                                     | Automatically create an admin account with this password. Requires `ADMIN_EMAIL`.                                                                                                                                                |
| CHECK_FOR_UPDATES             | bool        | true                                     | Check and notify users about new versions.  Checks GitHub API every 12 hours.                                                                                                                                                    |

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) login via `django-allauth`.  Users can authenticate using external OIDC providers.

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider. will be displayed in the login page. Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

Your OIDC provider needs a callback URL. For WYGIWYH, it's:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH URL. Replace `<OIDC_CLIENT_NAME>` with the slugified version of `OIDC_CLIENT_NAME`, or `openid-connect` if it's not set.

## How it Works

For more details, see the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Translate

Help translate WYGIWYH!

[![Translation Status](https://translations.herculino.com/widget/wygiwyh/open-graph.png)](https://translations.herculino.com/engage/wygiwyh/)

>   **Note:** Login with your GitHub account.

## Caveats and Warnings

*   I am not an accountant; calculations and terms might be inaccurate.  Please report any issues.
*   Calculations are mostly done at runtime, which can impact performance.
*   This is not a budgeting or double-entry accounting application.

## Built With

WYGIWYH uses these open-source tools:

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