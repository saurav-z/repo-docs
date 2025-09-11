# WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker

**[WYGIWYH](https://github.com/eitchtee/WYGIWYH) is a straightforward, open-source finance tracker built on the principle of "What You Get Is What You Have," empowering you to manage your money with clarity and flexibility.**

<img src="./.github/img/logo.png" alt="WYGIWYH Logo" width="200">

## Key Features:

*   **Unified Transaction Tracking:** Easily record all your income and expenses in one place.
*   **Multi-Account Support:** Monitor your funds across various accounts (banks, wallets, investments, etc.).
*   **Built-in Multi-Currency Support:** Seamlessly handle transactions and balances in different currencies.
*   **Custom Currencies:** Create and track custom currencies like crypto or reward points.
*   **Automated Transaction Rules:** Automate modifications to transactions using customizable rules.
*   **Dollar-Cost Averaging (DCA) Tracker:** Track your recurring investments for crypto, stocks, and more.
*   **API Support for Automation:** Integrate WYGIWYH with other services to automate transaction synchronization.

## Why WYGIWYH?

Tired of complex budgeting apps? WYGIWYH simplifies finance management by focusing on a core principle: **"Use what you earn this month for this month."** This approach helps you avoid dipping into savings while providing clear visibility into your spending.

## Demo

Explore WYGIWYH's functionality with a live demo:

*   **Demo Link:** [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)
*   **Credentials:**
    *   Email: `demo@demo.com`
    *   Password: `wygiwyhdemo`

>   **Important:**  Demo data is reset regularly. Automation features are disabled.

## Getting Started

WYGIWYH is a self-hosted application that utilizes Docker and Docker Compose.

**Prerequisites:**

*   [Docker](https://docs.docker.com/engine/install/)
*   [docker-compose](https://docs.docker.com/compose/install/)

**Installation Steps:**

1.  Create a project directory:
    ```bash
    mkdir WYGIWYH
    cd WYGIWYH
    ```
2.  Create and populate the `docker-compose.yml` file. See the [project's GitHub repository](https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml) for the complete example and edit according to your needs.
3.  Create and populate the `.env` file with your configurations. Use the [`.env.example`](https://github.com/eitchtee/WYGIWYH/blob/main/.env.example) as a starting point.
4.  Run the application:
    ```bash
    docker compose up -d
    ```
5.  Create the first admin account (optional, but recommended):
    ```bash
    docker compose exec -it web python manage.py createsuperuser
    ```

>   **Note:** For Unraid users, the WYGIWYH is available in the Unraid Store. Be sure to check the Unraid section below for further instructions.

## Running Locally

To run WYGIWYH locally, adjust your `.env` file:

1.  Remove `URL`.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 \[::1]).

You can then access the application at `localhost:OUTBOUND_PORT`.

>   **Tip:** If you're using a service like Tailscale, also include your machine's IP in `DJANGO_ALLOWED_HOSTS`.

## Latest Changes

For the latest updates and features, consider using the `:nightly` Docker tag or building from source.

## Unraid

WYGIWYH is available in the Unraid store. Please note that you will be required to provision your own postgres (version 15 or up) database. After installing the app, access the container's console via the Unraid UI and run `python manage.py createsuperuser` to create your first user.

## Environment Variables

Configure WYGIWYH using the following environment variables:

| Variable                       | Type        | Default                        | Description                                                                                                                                                                                                                                                                                                                        |
| ------------------------------ | ----------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DJANGO_ALLOWED_HOSTS`         | string      | localhost 127.0.0.1            | Specifies a list of allowed host/domain names for the WYGIWYH site.  More details in the [Django documentation](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts).                                                                                                                                            |
| `HTTPS_ENABLED`                | true\|false | false                          | Enables secure cookies, enforcing the "secure" flag for HTTPS connections.                                                                                                                                                                                                                                                      |
| `URL`                          | string      | http://localhost http://127.0.0.1 | A list of trusted origins for unsafe requests.  See the [Django documentation](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins) for further details.                                                                                                                                               |
| `SECRET_KEY`                   | string      | ""                             | The secret key used for cryptographic signing (essential for security).  Generate a unique and unpredictable value.                                                                                                                                                                                                            |
| `DEBUG`                        | true\|false | false                          | Enables debug mode (for development purposes only).                                                                                                                                                                                                                                                                               |
| `SQL_DATABASE`                 | string      | None (required)                | The name of your PostgreSQL database.                                                                                                                                                                                                                                                                                            |
| `SQL_USER`                     | string      | user                           | Your PostgreSQL database username.                                                                                                                                                                                                                                                                                                 |
| `SQL_PASSWORD`                 | string      | password                       | Your PostgreSQL database password.                                                                                                                                                                                                                                                                                                 |
| `SQL_HOST`                     | string      | localhost                      | Your PostgreSQL database host address.                                                                                                                                                                                                                                                                                             |
| `SQL_PORT`                     | string      | 5432                           | Your PostgreSQL database port.                                                                                                                                                                                                                                                                                                   |
| `SESSION_EXPIRY_TIME`          | int         | 2678400 (31 days)              | The duration (in seconds) before session cookies expire.                                                                                                                                                                                                                                                                          |
| `ENABLE_SOFT_DELETE`           | true\|false | false                          | Enables soft deletion of transactions, retaining them in the database (useful for imports and data management).                                                                                                                                                                                                                |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | 365                            | Time in days to retain soft-deleted transactions if `ENABLE_SOFT_DELETE` is true. If set to `0`, trasactions will be kept indefinitely.                                                                                                                                                                                          |
| `TASK_WORKERS`                 | int         | 1                              | Number of workers to use for asynchronous tasks.                                                                                                                                                                                                                                                                                  |
| `DEMO`                         | true\|false | false                          | Enables demo mode.                                                                                                                                                                                                                                                                                                                 |
| `ADMIN_EMAIL`                  | string      | None                           | Automatically creates an admin account with the specified email. Requires `ADMIN_PASSWORD` to be set as well.                                                                                                                                                                                                                    |
| `ADMIN_PASSWORD`               | string      | None                           | Sets an admin password, used in conjunction with `ADMIN_EMAIL`.                                                                                                                                                                                                                                                                    |
| `CHECK_FOR_UPDATES`            | bool        | true                           | Determines if the application checks and notifies users of new version releases by polling Github's API every 12 hours.                                                                                                                                                                                                 |

## OIDC Configuration

Configure OIDC for login using these environment variables:

| Variable             | Description                                                                                                                                                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | Provider name that will appear in the login page. Defaults to `OpenID Connect`.                                                                                                                                                              |
| `OIDC_CLIENT_ID`     | Client ID provided by your OIDC provider.                                                                                                                                                                                                    |
| `OIDC_CLIENT_SECRET` | Client Secret provided by your OIDC provider.                                                                                                                                                                                                |
| `OIDC_SERVER_URL`    | OIDC provider's base URL for discovery or the authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`).  Used by `django-allauth`.                                                                                     |
| `OIDC_ALLOW_SIGNUP`  | Allows automatic creation of accounts upon successful authentication.  Defaults to `true`.                                                                                                                                                     |

**Callback URL (Redirect URI):**

The redirect URI for your OIDC provider should be:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH instance's URL and `<OIDC_CLIENT_NAME>` with the sluggified value set in OIDC_CLIENT_NAME, or the default value, openid-connect if you haven't set this variable.

## Contribute to WYGIWYH Translations!

Help localize WYGIWYH into your language by visiting:
<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>
>   **Note:**  Login with your GitHub account to contribute.

## Important Considerations

*   **Disclaimer:** WYGIWYH is not created by a financial expert; open issues if you spot any calculations or terms you feel require attention.
*   **Performance:** Most calculations occur at runtime which may cause some performance degradation on larger transaction sets.
*   **Scope:** WYGIWYH doesn't offer budgeting features.

## Built With

WYGIWYH is built with the following open-source tools:

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