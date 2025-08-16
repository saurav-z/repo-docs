# WYGIWYH: Take Control of Your Finances with a Simple Approach

**WYGIWYH** (What You Get Is What You Have) is a finance tracker designed for users who prioritize simplicity and a straightforward, no-budgeting approach. [Explore the WYGIWYH repository](https://github.com/eitchtee/WYGIWYH) to start managing your money more effectively!

*   **Streamlined Finance Tracking:** Simplify your finances with unified income and expense tracking.
*   **Multi-Currency Support:** Easily manage transactions and balances in multiple currencies.
*   **Automated Features:** Leverage custom transaction rules and a built-in Dollar-Cost Averaging (DCA) tracker.
*   **Flexible & Customizable:** Tailor the system to your needs with custom currencies and API support.
*   **Open-Source & Self-Hosted:** Maintain control over your financial data with a self-hosted solution.

<img src="./.github/img/monthly_view.png" width="18%">
<img src="./.github/img/yearly.png" width="18%">
<img src="./.github/img/networth.png" width="18%">
<img src="./.github/img/calendar.png" width="18%">
<img src="./.github/img/all_transactions.png" width="18%">

## Key Features

WYGIWYH offers powerful features to simplify and streamline your personal finance management:

*   **Unified Transaction Tracking:** Record all your income and expenses in one place.
*   **Multi-Account Support:** Track your finances across various accounts (banks, wallets, investments, etc.).
*   **Built-in Multi-Currency Support:** Manage transactions and balances seamlessly in different currencies.
*   **Custom Currencies:** Create your own currencies for crypto, rewards points, or any model you need.
*   **Automated Adjustments with Rules:** Automatically modify transactions using customizable rules.
*   **Dollar-Cost Average (DCA) Tracker:** Essential for tracking recurring investments.
*   **API Support:** Integrate with other tools and services for automated transaction synchronization.

## Why WYGIWYH?

WYGIWYH is designed for a simple, no-budgeting approach to finance management based on the principle:

> Use what you earn this month for this month. Any savings are tracked but treated as untouchable for future months.

This principle helps you avoid overspending while still tracking your financial activity.

## Demo

Experience WYGIWYH firsthand: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

*   **Email:** `demo@demo.com`
*   **Password:** `wygiwyhdemo`

> [!NOTE]
> Any data you add will be wiped within 24 hours.

## How To Use

WYGIWYH is designed to be deployed using Docker and Docker Compose.

1.  **Prerequisites:** Ensure you have [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed.
2.  **Get the Code:**  Clone or download the WYGIWYH repository.
3.  **Configure:** Create a `.env` file based on the example `.env.example` and customize the settings.
4.  **Run:** Execute `docker compose up -d` in your terminal to start the application.
5.  **Admin Account:**  Use `docker compose exec -it web python manage.py createsuperuser` (or configure `ADMIN_EMAIL` and `ADMIN_PASSWORD` in your `.env` file).

### Running Locally

1.  In `.env`, remove `URL` and set `HTTPS_ENABLED=false`.
2.  Leave the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).
3.  Access the application via `localhost:OUTBOUND_PORT`.

    > [!NOTE]
    > If you're using Tailscale, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.

### Latest Changes

For the latest updates, build from source or use the `:nightly` Docker tag.

### Unraid

WYGIWYH is also available in the Unraid Community Applications. You'll need to provision your own PostgreSQL database (version 15 or up).

## Environment Variables

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                                |
|-------------------------------|-------------|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | A list of space-separated domains and IPs representing the host/domain names that WYGIWYH can serve.                                                                                                                                   |
| HTTPS_ENABLED                 | true\|false | false                             | Whether to use secure cookies. If set to true, the cookie will be marked as "secure."                                                                                                                                                     |
| URL                           | string      | http://localhost http://127.0.0.1 | A list of space-separated domains and IPs (with protocol) representing the trusted origins for unsafe requests (e.g., POST).                                                                                                          |
| SECRET_KEY                    | string      | ""                                |  Used for cryptographic signing.  Must be a unique, unpredictable value.                                                                                                                                                                    |
| DEBUG                         | true\|false | false                             | Turns DEBUG mode on or off.  Use this in production for more data about possible errors.                                                                                                                                                  |
| SQL_DATABASE                  | string      | None *required                    | The name of your PostgreSQL database.                                                                                                                                                                                                      |
| SQL_USER                      | string      | user                              | The username for your PostgreSQL database.                                                                                                                                                                                                 |
| SQL_PASSWORD                  | string      | password                          | The password for your PostgreSQL database.                                                                                                                                                                                                 |
| SQL_HOST                      | string      | localhost                         | The address for your PostgreSQL database.                                                                                                                                                                                                  |
| SQL_PORT                      | string      | 5432                              | The port for your PostgreSQL database.                                                                                                                                                                                                     |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                 | The age of session cookies, in seconds.                                                                                                                                                                                                  |
| ENABLE_SOFT_DELETE            | true\|false | false                             | Enable or disable soft deletes for transactions. Deleted transactions will remain in the database if enabled. Useful for imports and avoiding duplicate entries.                                                                          |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                               | Time in days to keep soft-deleted transactions. If 0, will keep all transactions indefinitely. Only works if ENABLE_SOFT_DELETE is true.                                                                                                  |
| TASK_WORKERS                  | int         | 1                                 | The number of workers for async tasks.                                                                                                                                                                                                    |
| DEMO                          | true\|false | false                             | Enables demo mode.                                                                                                                                                                                                                        |
| ADMIN_EMAIL                   | string      | None                              | Automatically creates an admin account with this email. Must also have `ADMIN_PASSWORD` set.                                                                                                                                              |
| ADMIN_PASSWORD                | string      | None                              | Automatically creates an admin account with this password. Must also have `ADMIN_EMAIL` set.                                                                                                                                              |
| CHECK_FOR_UPDATES             | bool        | true                              | Check and notify users about new versions. The check is done by doing a single query to Github's API every 12 hours.                                                                                                                   |

## OIDC Configuration

Configure OpenID Connect (OIDC) login for user authentication via `django-allauth`.

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider. will be displayed in the login page. Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**
`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` and `<OIDC_CLIENT_NAME>` with the proper values for your deployment.

## How it Works

Learn more about WYGIWYH on the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Help Us Translate!

Contribute to the project's localization efforts:
<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

## Caveats and Warnings

*   I'm not an accountant, ensure to open an issue if you see anything that could be improved.
*   Calculations happen at runtime, which can cause some performance degradation.
*   This is not a budgeting or double-entry accounting application.

## Built With

WYGIWYH is built upon several open-source tools:

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