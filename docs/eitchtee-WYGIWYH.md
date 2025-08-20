<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with Simplicity
  <br>
</h1>

<h4 align="center">An opinionated and powerful finance tracker.</h4>

<p align="center">
  <a href="#why-wygiwyh">Why</a> •
  <a href="#key-features">Features</a> •
  <a href="#how-to-use">Usage</a> •
  <a href="#how-it-works">How</a> •
  <a href="#help-us-translate-wygiwyh">Translate</a> •
  <a href="#caveats-and-warnings">Caveats and Warnings</a> •
  <a href="#built-with">Built with</a>
  <br>
  <a href="https://github.com/eitchtee/WYGIWYH">🔗 View the Original Repository</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a straightforward finance tracker designed for those who prefer a no-budget, "use it this month, for this month" approach to money management.  This finance tracker focuses on simplicity and flexibility.

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## Why WYGIWYH?

Tired of complex budgeting apps? WYGIWYH (pronounced "wiggy-wih") simplifies finances with a core principle: *Use what you earn this month for this month.*  Savings are tracked but considered untouchable for future months.

This philosophy simplifies financial tracking, but finding suitable tools was challenging.  WYGIWYH addresses the need for:

1.  **Multi-currency Support:** Manage income and expenses in different currencies.
2.  **Budget-Free:** Avoid the constraints of traditional budgeting.
3.  **Web App Usability:** Accessible and user-friendly, with optional mobile support.
4.  **Automation-Ready API:** Integrate with other tools and services.
5.  **Custom Transaction Rules:** Handle credit card billing cycles and other financial complexities.

WYGIWYH was born out of the need for a flexible and powerful tool that could handle these requirements.

## Key Features

**WYGIWYH** offers powerful features to streamline your personal finance tracking:

*   **Unified Transaction Tracking:**  One place to record all income and expenses.
*   **Multiple Account Support:** Track funds across banks, wallets, and investments.
*   **Multi-Currency Support:** Manage transactions and balances in various currencies.
*   **Custom Currencies:** Create currencies for crypto, rewards points, etc.
*   **Automated Rules:** Automatically adjust transactions using customizable rules.
*   **Dollar-Cost Average (DCA) Tracker:** Track recurring investments for crypto and stocks.
*   **API Support:** Seamlessly integrate with existing services to automate transactions.

## Demo

Test drive WYGIWYH at [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/) using:

> [!NOTE]
> E-mail: `demo@demo.com`
>
> Password: `wygiwyhdemo`

Note: Demo data is wiped regularly, and advanced features are disabled.

## How To Use

WYGIWYH requires [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).

1.  **Create a folder (optional):**

```bash
mkdir WYGIWYH
cd WYGIWYH
```

2.  **Create and Edit `docker-compose.yml`:**

```bash
touch docker-compose.yml
nano docker-compose.yml
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs
```

3.  **Create and Edit `.env`:**

```bash
touch .env
nano .env # or any other editor you want to use
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly
```

4.  **Run the app:**

```bash
docker compose up -d
```

5.  **Create an Admin Account (if not set in environment variables):**

```bash
docker compose exec -it web python manage.py createsuperuser
```

> [!NOTE]
> Unraid users: See the [Unraid section](#unraid).

### Running Locally

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

Access the app at `localhost:OUTBOUND_PORT`.

> [!NOTE]
> If using Tailscale or similar, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.  For other IPs, also add them to `DJANGO_ALLOWED_HOSTS`, without `http://`.

### Latest Changes

Features are added to `main` when ready.  Use the `:nightly` tag on Docker for the latest, potentially unstable, version.

Required Dockerfiles are available [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

## Unraid

[nwithan8](https://github.com/nwithan8) provides an Unraid template. Find it in the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is also available on the Unraid Store.  You'll need your own PostgreSQL database (version 15 or up).

Create the first user via the container's console in the Unraid UI: `python manage.py createsuperuser`.

## Environment Variables

| Variable                      | Type        | Default                          | Description                                                                                                                                                                                                                                                                                          |
| :---------------------------- | :---------- | :------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1              | Space-separated domains/IPs that the WYGIWYH site can serve.  See [Django documentation](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts) for details.                                                                                                                          |
| HTTPS_ENABLED                 | true\|false | false                            | Enables secure cookies.  Set to `true` for HTTPS.                                                                                                                                                                                                                                                         |
| URL                           | string      | http://localhost http://127.0.0.1 | Space-separated domains/IPs (with protocol) for trusted origins for unsafe requests (e.g., POST).  See [Django documentation](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins).                                                                                         |
| SECRET_KEY                    | string      | ""                               | Cryptographic signing key; set to a unique, unpredictable value.                                                                                                                                                                                                                                           |
| DEBUG                         | true\|false | false                            | Enable or disable DEBUG mode.  Useful for debugging; do *not* use in production.                                                                                                                                                                                                                       |
| SQL_DATABASE                  | string      | None *required*                  | PostgreSQL database name.                                                                                                                                                                                                                                                                              |
| SQL_USER                      | string      | user                             | PostgreSQL username.                                                                                                                                                                                                                                                                                     |
| SQL_PASSWORD                  | string      | password                         | PostgreSQL password.                                                                                                                                                                                                                                                                                     |
| SQL_HOST                      | string      | localhost                        | PostgreSQL host address.                                                                                                                                                                                                                                                                               |
| SQL_PORT                      | string      | 5432                             | PostgreSQL port.                                                                                                                                                                                                                                                                                       |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                | Session cookie age in seconds (how long users stay logged in).                                                                                                                                                                                                                                       |
| ENABLE_SOFT_DELETE            | true\|false | false                            | Enable soft deletes for transactions. Deleted transactions are retained in the database. Useful for imports and avoiding duplicates.                                                                                                                                                                |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                              | Days to keep soft-deleted transactions.  `0` keeps them indefinitely. Only works if `ENABLE_SOFT_DELETE` is `true`.                                                                                                                                                                                 |
| TASK_WORKERS                  | int         | 1                                | Number of workers for async tasks.                                                                                                                                                                                                                                                                   |
| DEMO                          | true\|false | false                            | Enables demo mode.                                                                                                                                                                                                                                                                                       |
| ADMIN_EMAIL                   | string      | None                             | Creates an admin account with this email. Requires `ADMIN_PASSWORD` to also be set.                                                                                                                                                                                                                     |
| ADMIN_PASSWORD                | string      | None                             | Creates an admin account with this password. Requires `ADMIN_EMAIL` to also be set.                                                                                                                                                                                                                    |
| CHECK_FOR_UPDATES             | bool        | true                             | Check and notify users about new versions. The check is done by doing a single query to Github's API every 12 hours.                                                                                                                                                                                              |

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) for login via `django-allauth`.

> [!NOTE]
> Currently only OpenID Connect is supported as a provider, open an issue if you need something else.

Configure OIDC using the following environment variables:

| Variable             | Description                                                                                                                                                                                                                                            |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | Provider name (displayed in the login page). Defaults to `OpenID Connect`.                                                                                                                                                                              |
| `OIDC_CLIENT_ID`     | The Client ID from your OIDC provider.                                                                                                                                                                                                                   |
| `OIDC_CLIENT_SECRET` | The Client Secret from your OIDC provider.                                                                                                                                                                                                               |
| `OIDC_SERVER_URL`    | OIDC provider's discovery document or authorization server URL (e.g., `https://your-provider.com/auth/realms/your-realm`).  `django-allauth` uses this to find endpoints.                                                                             |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic account creation on successful authentication. Defaults to `true`.                                                                                                                                                                         |

**Callback URL (Redirect URI):**

When setting up your OIDC provider, use the following callback URL (Redirect URI):

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` and `<OIDC_CLIENT_NAME>` with the actual values for your instance.

## How it Works

Learn more in the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Help Us Translate WYGIWYH!

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your GitHub account.

## Caveats and Warnings

*   I'm not an accountant; some terms or calculations might be inaccurate.  Report issues.
*   Most calculations are performed at runtime, which may impact performance.
*   This is *not* a budgeting or double-entry accounting application.

## Built With

WYGIWYH relies on great open-source tools, including:

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
```
Key Improvements and Summary:

*   **SEO Optimization:** Added a strong, keyword-rich title and a one-sentence hook at the beginning to capture attention.
*   **Structure and Readability:** Improved the overall structure by using headings, bullet points, and better formatting.
*   **Key Feature Highlighting:** Features are now more clearly outlined and easier to understand.
*   **Conciseness:** Removed redundant phrases and streamlined the text.
*   **Clear Instructions:** Usage instructions are clear and include Unraid-specific notes.
*   **Environment Variable Table:** Improved clarity and information about environment variables using a table.
*   **Translation and Contribution:** Emphasized the importance of translation and where to contribute.
*   **Links:** Added a link back to the original repo for improved SEO.
*   **Concise Warnings:** Summarized the warnings for better understanding.