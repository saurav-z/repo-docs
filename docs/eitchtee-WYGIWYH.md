<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Intuitive Finance Tracker
  <br>
</h1>

<h4 align="center">Take control of your finances with a straightforward, no-budget approach.</h4>

<p align="center">
  <a href="#about">About</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#translations">Translations</a> •
  <a href="#caveats">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a powerful, open-source finance tracker designed for users who prefer a simple, no-budget approach to money management. This tool prioritizes clarity and ease of use, helping you understand where your money goes. Track income, expenses, and investments with multi-currency support and customizable features.

<img src=".github/img/monthly_view.png" width="18%"></img> <img src=".github/img/yearly.png" width="18%"></img> <img src=".github/img/networth.png" width="18%"></img> <img src=".github/img/calendar.png" width="18%"></img> <img src=".github/img/all_transactions.png" width="18%"></img>

## About

WYGIWYH simplifies money management by focusing on a core principle: *Use what you earn this month for this month.* This approach emphasizes tracking and understanding your spending habits without the constraints of a budget. WYGIWYH was built out of frustration with existing financial tools, addressing the need for multi-currency support, customizable transaction rules, and API integration for automation.

## Key Features

WYGIWYH offers a comprehensive set of features to streamline your personal finance tracking:

*   **Unified Transaction Tracking:** Easily record and organize all income and expenses in one place.
*   **Multiple Account Support:** Track your finances across various accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:** Seamlessly manage transactions and balances in different currencies.
*   **Custom Currencies:** Define your own currencies for crypto, rewards points, or other needs.
*   **Automated Adjustments with Rules:** Automatically modify transactions based on customizable rules.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Track recurring investments with ease, perfect for crypto and stocks.
*   **API Support for Automation:** Integrate with other services to automatically synchronize transactions.

## Demo

Explore WYGIWYH's capabilities with our live demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

Use the following credentials:

>   [!NOTE]
>   E-mail: `demo@demo.com`
>
>   Password: `wygiwyhdemo`

**Please note:** Data in the demo is reset frequently, usually within 24 hours. Automation features (API, Rules, Exchange Rates, Import/Export) are disabled in the demo.

## Getting Started

WYGIWYH uses [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) for deployment.

```bash
# Create a project directory (optional)
$ mkdir WYGIWYH
$ cd WYGIWYH

# Create docker-compose.yml
$ touch docker-compose.yml
$ nano docker-compose.yml
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs

# Create .env file
$ touch .env
$ nano .env
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly

# Run the application
$ docker compose up -d

# Create the first admin account (Optional if using ADMIN_EMAIL and ADMIN_PASSWORD)
$ docker compose exec -it web python manage.py createsuperuser
```

>   [!NOTE]
>   If using Unraid, use the app from the store; See the [Unraid section](#unraid) and [Environment Variables](#environment-variables) for configuration details.

### Running Locally

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Leave the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1])
4.  Access the application at `localhost:OUTBOUND_PORT`.

>   [!NOTE]
>   If using services like Tailscale, add your machine's IP to `DJANGO_ALLOWED_HOSTS`. For non-localhost access, add the IP to `DJANGO_ALLOWED_HOSTS` without `http://`.

## Configuration

### Environment Variables

Customize WYGIWYH using these environment variables:

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                             |
| :---------------------------- | :---------- | :-------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DJANGO_ALLOWED_HOSTS`          | string      | `localhost 127.0.0.1`               |  Domains/IPs that the site can serve. [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                                                                                                       |
| `HTTPS_ENABLED`                 | true\|false | `false`                             |  Enable secure cookies.                                                                                                                                                                                                          |
| `URL`                           | string      | `http://localhost http://127.0.0.1` | Trusted origins for unsafe requests. [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                                                                                     |
| `SECRET_KEY`                    | string      | `""`                              | Cryptographic signing key; use a unique, unpredictable value.                                                                                                                                                                         |
| `DEBUG`                         | true\|false | `false`                             | Enable debug mode (don't use in production).                                                                                                                                                                                         |
| `SQL_DATABASE`                  | string      | *Required*                        | Name of your PostgreSQL database.                                                                                                                                                                                                       |
| `SQL_USER`                      | string      | `user`                            | PostgreSQL username.                                                                                                                                                                                                                    |
| `SQL_PASSWORD`                  | string      | `password`                        | PostgreSQL password.                                                                                                                                                                                                                    |
| `SQL_HOST`                      | string      | `localhost`                       | PostgreSQL host address.                                                                                                                                                                                                                |
| `SQL_PORT`                      | string      | `5432`                            | PostgreSQL port.                                                                                                                                                                                                                        |
| `SESSION_EXPIRY_TIME`           | int         | `2678400` (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                     |
| `ENABLE_SOFT_DELETE`            | true\|false | `false`                             | Enable soft-deletion of transactions.                                                                                                                                                                                                 |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | `365`                             | Time in days to keep soft-deleted transactions (works if `ENABLE_SOFT_DELETE` is true).                                                                                                                                                  |
| `TASK_WORKERS`                  | int         | `1`                                 | Number of workers for asynchronous tasks.                                                                                                                                                                                           |
| `DEMO`                          | true\|false | `false`                             | Enable demo mode.                                                                                                                                                                                                                    |
| `ADMIN_EMAIL`                   | string      | `None`                            | Creates an admin account with the specified email (requires `ADMIN_PASSWORD`).                                                                                                                                                             |
| `ADMIN_PASSWORD`                | string      | `None`                            | Creates an admin account with the specified password (requires `ADMIN_EMAIL`).                                                                                                                                                           |
| `CHECK_FOR_UPDATES`             | bool        | `true`                              | Check for new version notifications (queries GitHub API every 12 hours).                                                                                                                                               |

### OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) for authentication.

| Variable             | Description                                                                                                                                                                                                                                                                           |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | The provider name; shown on the login page. Defaults to `OpenID Connect`.                                                                                                                                                                                                              |
| `OIDC_CLIENT_ID`     | Client ID from your OIDC provider.                                                                                                                                                                                                                                                      |
| `OIDC_CLIENT_SECRET` | Client Secret from your OIDC provider.                                                                                                                                                                                                                                                  |
| `OIDC_SERVER_URL`    | Base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`).  `django-allauth` will use this to discover the necessary endpoints.                                                                                             |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic account creation after successful authentication. Defaults to `true`.                                                                                                                                                                                                |

**Callback URL (Redirect URI):**
Set the callback URL (Redirect URI) in your OIDC provider's settings to:
`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`
Replace `https://your.wygiwyh.domain` and `<OIDC_CLIENT_NAME>` accordingly.

### Unraid

WYGIWYH is available in the Unraid Store. You'll need to set up your own PostgreSQL database (version 15 or higher).

To create the first user, open the container's console and run `python manage.py createsuperuser`.

## How It Works

Learn more about WYGIWYH's concepts and features in the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Translations

Contribute to WYGIWYH's internationalization by visiting:  [Translation Portal](https://translations.herculino.com/engage/wygiwyh/)

>   [!NOTE]
>   Log in with your GitHub account to contribute.

## Caveats & Warnings

*   I'm not an accountant; some terms and calculations might be inaccurate. Please open an issue if you find errors.
*   Most calculations are done at runtime, which may impact performance.
*   WYGIWYH is not a budgeting or double-entry accounting app.

## Built With

WYGIWYH is built with several open-source technologies:

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