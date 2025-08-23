# WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker

**[Visit the WYGIWYH Repository on GitHub](https://github.com/eitchtee/WYGIWYH)**

WYGIWYH (What You Get Is What You Have) is a modern, open-source finance tracker designed for users who prefer a straightforward, no-budget approach to money management.  This application simplifies personal finance by providing essential features within a flexible framework.

*   **Unified Transaction Tracking:** Easily record and organize all income and expenses in one place.
*   **Multi-Account Support:** Track funds across various accounts, including banks, wallets, and investments.
*   **Multi-Currency Support:**  Effortlessly manage transactions and balances in different currencies.
*   **Customizable Currencies:** Create custom currencies for tracking crypto, rewards points, or other financial models.
*   **Automated Transaction Rules:** Set up rules to automatically adjust transactions.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Simplify tracking recurring investments.
*   **API Support for Automation:** Integrate with other services and automate your financial workflows.

---

## Why Choose WYGIWYH?

Tired of overly complex budgeting apps?  WYGIWYH is built on a simple principle: **Use what you earn this month for this month, treating savings as untouchable for future months.** This clear approach allows for efficient tracking of your money without the restrictions of traditional budgeting.  WYGIWYH addresses the need for a flexible, multi-currency, and automated finance tracking tool.

---

## Key Features

*   **Intuitive Interface:** A clean, user-friendly interface for easy navigation and data entry.
*   **Comprehensive Reporting:** Track your finances with detailed monthly, yearly, and net worth views, as well as a calendar overview.
*   **Customizable Dashboard:** Configure the dashboard to show the most relevant financial information at a glance.
*   **Automated Exchange Rates:** (Future Feature)  Automatically update currency exchange rates to ensure accurate conversions.
*   **Import/Export Functionality:** (Future Feature) Easily import and export transaction data for backup and integration.

---

## Demo

Explore WYGIWYH's features with our interactive demo:

*   **Demo URL:** [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)
*   **Email:** `demo@demo.com`
*   **Password:** `wygiwyhdemo`

**Note:**  Demo data is reset every 24 hours.  API, Rules, Automatic Exchange Rates, and Import/Export features are disabled in the demo.

---

## Getting Started

WYGIWYH is designed to be run using Docker and Docker Compose.

1.  **Prerequisites:** Ensure you have [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed.
2.  **Configuration:**  Follow the instructions in the original [README](https://github.com/eitchtee/WYGIWYH) to set up your `.env` file with your database and security settings and to set up `docker-compose.yml`.  A `docker-compose.prod.yml` example is also available.
3.  **First Run:** Execute `docker compose up -d` to start the application.  Create your first admin account via `docker compose exec -it web python manage.py createsuperuser`.

### Running Locally

To run WYGIWYH locally:

1.  In your `.env` file, remove the `URL` setting.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` setting.  Access the application via `localhost:OUTBOUND_PORT`.

### Unraid

WYGIWYH is available via the Unraid App Store. Follow the specific instructions provided in the store.  You will need to provision your own PostgreSQL database (version 15 or higher).

### Environment Variables

WYGIWYH is highly configurable via environment variables:

| Variable                      | Type        | Default                          | Description                                                                                                                                                                                                                                                                                                      |
|-------------------------------|-------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DJANGO_ALLOWED_HOSTS`          | string      | localhost 127.0.0.1              | A list of space-separated domains and IPs representing the host/domain names that WYGIWYH can serve.                                                                                                                                                                                                             |
| `HTTPS_ENABLED`                 | true\|false | false                             | Whether to use secure cookies. Set to `true` for HTTPS connections.                                                                                                                                                                                                                                               |
| `URL`                           | string      | http://localhost http://127.0.0.1 | A list of space-separated domains and IPs (with the protocol) representing the trusted origins for unsafe requests (e.g., POST).                                                                                                                                                                                   |
| `SECRET_KEY`                    | string      | ""                                | Your unique cryptographic signing key.                                                                                                                                                                                                                                                                             |
| `DEBUG`                         | true\|false | false                             | Enables or disables debug mode.  **Do not use in production.**                                                                                                                                                                                                                                                      |
| `SQL_DATABASE`                  | string      | None *required*                 | The name of your PostgreSQL database.                                                                                                                                                                                                                                                                           |
| `SQL_USER`                      | string      | user                              | PostgreSQL database username.                                                                                                                                                                                                                                                                                   |
| `SQL_PASSWORD`                  | string      | password                          | PostgreSQL database password.                                                                                                                                                                                                                                                                                   |
| `SQL_HOST`                      | string      | localhost                         | PostgreSQL database host address.                                                                                                                                                                                                                                                                                 |
| `SQL_PORT`                      | string      | 5432                              | PostgreSQL database port.                                                                                                                                                                                                                                                                                       |
| `SESSION_EXPIRY_TIME`           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                                                                                               |
| `ENABLE_SOFT_DELETE`            | true\|false | false                             | Enables soft deletion of transactions.                                                                                                                                                                                                                                                                             |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | 365                               | The time (in days) to keep soft-deleted transactions.  Only applies if `ENABLE_SOFT_DELETE` is true.                                                                                                                                                                                                              |
| `TASK_WORKERS`                  | int         | 1                                 | Number of workers for asynchronous tasks.                                                                                                                                                                                                                                                                         |
| `DEMO`                          | true\|false | false                             | Enables demo mode.                                                                                                                                                                                                                                                                                                |
| `ADMIN_EMAIL`                   | string      | None                              | Automatically creates an admin account with this email if `ADMIN_PASSWORD` is also set.                                                                                                                                                                                                                         |
| `ADMIN_PASSWORD`                | string      | None                              | Automatically creates an admin account with this password if `ADMIN_EMAIL` is also set.                                                                                                                                                                                                                         |
| `CHECK_FOR_UPDATES`             | bool        | true                              | Check for and notify users about new versions. The check is done by doing a single query to Github's API every 12 hours.                                                                                                                                                                                       |

### OIDC Configuration

WYGIWYH supports login through OpenID Connect (OIDC) using `django-allauth`:

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider. will be displayed in the login page. Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

Configure your OIDC provider with the following callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH instance's URL and `<OIDC_CLIENT_NAME>` with the `OIDC_CLIENT_NAME` you chose or the default value of `openid-connect`.

---

## How It Works

For more in-depth information, please consult the [WYGIWYH Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

---

## Contribute & Translate

Help improve WYGIWYH!  Contribute to the project or help with translations via the [translation portal](https://translations.herculino.com/engage/wygiwyh/).

---

## Important Considerations

*   **Disclaimer:**  I am not an accountant.  Please report any inaccuracies.
*   **Performance:**  Calculations are done at runtime, which may impact performance.
*   **Focus:**  WYGIWYH is not a budgeting or double-entry accounting application.

---

## Built With

WYGIWYH is built with a powerful stack of open-source technologies, including:

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