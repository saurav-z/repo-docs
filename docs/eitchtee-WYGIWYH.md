# WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker

**WYGIWYH** (What You Get Is What You Have) is a flexible and user-friendly finance tracker designed to help you manage your money without the constraints of budgeting. This open-source project provides a straightforward approach to track your income, expenses, and investments. [Explore WYGIWYH on GitHub](https://github.com/eitchtee/WYGIWYH).

## Key Features

*   **Unified Transaction Tracking:** Easily record all your income and expenses in one centralized location.
*   **Multi-Account Support:** Manage various accounts such as banks, wallets, and investments all in one place.
*   **Multi-Currency Capabilities:** Effortlessly track transactions and balances in different currencies.
*   **Custom Currencies:** Create and manage custom currencies for various assets, rewards, or other models.
*   **Automated Adjustments with Rules:** Automate transaction modifications based on customizable rules.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Track recurring investments for stocks and cryptocurrencies with ease.
*   **API Support for Automation:** Integrate with existing services to automate transactions and streamline data.

## Why WYGIWYH?

Frustrated with complex budgeting apps and the lack of flexible finance tracking tools, WYGIWYH was created to provide a simple, yet powerful solution. Based on the principle of "Use what you earn this month for this month," WYGIWYH helps you avoid overspending while still monitoring your financial health.

## Demo

Experience WYGIWYH firsthand! You can try a demo version at [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/) using the following credentials:

>   **E-mail:** demo@demo.com
>
>   **Password:** wygiwyhdemo

*Note: Data added to the demo will be wiped within 24 hours.*

## Getting Started

WYGIWYH utilizes Docker and docker-compose for easy setup:

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

To run WYGIWYH locally:

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

Then, access the application via `localhost:OUTBOUND_PORT`.

>   *   If running behind Tailscale or similar, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
>   *   For non-localhost IPs, add them to `DJANGO_ALLOWED_HOSTS`.

## Unraid

WYGIWYH is available on the Unraid Store. Users need to provision their own PostgreSQL database (version 15 or up).

To create the first user, open the container's console via the Unraid UI, and type `python manage.py createsuperuser`.

## Environment Variables

Configure WYGIWYH using the following environment variables:

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                             |
| ----------------------------- | ----------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DJANGO_ALLOWED_HOSTS`        | string      | localhost 127.0.0.1               | List of allowed domains and IPs.                                                                                                                                                                        |
| `HTTPS_ENABLED`               | true\|false | false                             | Enable/disable secure cookies.                                                                                                                                                                           |
| `URL`                         | string      | http://localhost http://127.0.0.1 | List of trusted origins for unsafe requests.                                                                                                                                                             |
| `SECRET_KEY`                  | string      | ""                                | Unique key for cryptographic signing.                                                                                                                                                                   |
| `DEBUG`                       | true\|false | false                             | Enable/disable debug mode (don't use in production).                                                                                                                                                      |
| `SQL_DATABASE`                | string      | None \*required                   | PostgreSQL database name.                                                                                                                                                                             |
| `SQL_USER`                    | string      | user                              | PostgreSQL username.                                                                                                                                                                                    |
| `SQL_PASSWORD`                | string      | password                          | PostgreSQL password.                                                                                                                                                                                    |
| `SQL_HOST`                    | string      | localhost                         | PostgreSQL host address.                                                                                                                                                                                |
| `SQL_PORT`                    | string      | 5432                              | PostgreSQL port.                                                                                                                                                                                        |
| `SESSION_EXPIRY_TIME`         | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                          |
| `ENABLE_SOFT_DELETE`          | true\|false | false                             | Enable/disable soft delete of transactions.                                                                                                                                                             |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | 365                               | Time (days) to keep soft-deleted transactions.                                                                                                                                                            |
| `TASK_WORKERS`                | int         | 1                                 | Number of workers for async tasks.                                                                                                                                                                        |
| `DEMO`                        | true\|false | false                             | Enable demo mode.                                                                                                                                                                                        |
| `ADMIN_EMAIL`                 | string      | None                              | Email for automatically creating an admin account (with `ADMIN_PASSWORD`).                                                                                                                                |
| `ADMIN_PASSWORD`              | string      | None                              | Password for automatically creating an admin account (with `ADMIN_EMAIL`).                                                                                                                               |
| `CHECK_FOR_UPDATES`           | bool        | true                              | Check and notify users about new versions (checks GitHub API every 12 hours).                                                                                                                             |

## OIDC Configuration

WYGIWYH supports login through OpenID Connect (OIDC) via `django-allauth`.

| Variable             | Description                                                                                                                                                                                                                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | Provider name (displayed in the login page). Defaults to `OpenID Connect`                                                                                                                                                                                                                             |
| `OIDC_CLIENT_ID`     | Your OIDC provider's Client ID.                                                                                                                                                                                                                                                                         |
| `OIDC_CLIENT_SECRET` | Your OIDC provider's Client Secret.                                                                                                                                                                                                                                                                     |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.).                                                     |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic account creation. Defaults to `true`.                                                                                                                                                                                                                                                      |

**Callback URL:**

Configure your OIDC provider with the callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` and `<OIDC_CLIENT_NAME>` with your actual values.

## How It Works

For further details, consult our [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Help Us Translate

Contribute to the localization of WYGIWYH!

[![Translation status](https://translations.herculino.com/widget/wygiwyh/open-graph.png)](https://translations.herculino.com/engage/wygiwyh/)

*Login with your GitHub account to contribute.*

## Caveats and Warnings

*   Consult an accountant for financial advice.
*   Some calculations may impact performance.
*   WYGIWYH isn't a budgeting or double-entry accounting application.

## Built With

WYGIWYH is developed with the help of these open-source tools:

*   Django
*   HTMX
*   \_hyperscript
*   Procrastinate
*   Bootstrap
*   Tailwind
*   Webpack
*   PostgreSQL
*   Django REST framework
*   Alpine.js