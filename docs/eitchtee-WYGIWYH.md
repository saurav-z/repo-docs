<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple Approach
  <br>
</h1>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#demo">Demo</a> •
  <a href="#how-it-works">How it Works</a> •
  <a href="#help-us-translate-wygiwyh">Translate</a> •
  <a href="#caveats-and-warnings">Caveats and Warnings</a> •
  <a href="#built-with">Built With</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a powerful, open-source finance tracker designed for a straightforward, no-budget approach to money management. **[Learn more and access the source code on GitHub](https://github.com/eitchtee/WYGIWYH).**

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## Key Features

*   **Unified Transaction Tracking:** Track all income and expenses in one place.
*   **Multi-Account Support:** Manage multiple bank accounts, wallets, and investment accounts.
*   **Multi-Currency Support:** Handle transactions and balances in various currencies.
*   **Custom Currency Options:** Create your own currencies for crypto, rewards, etc.
*   **Automated Transaction Rules:** Customize transaction rules for automation.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Monitor your recurring investments.
*   **API Integration:** Integrate seamlessly with other services for automation.

## Why WYGIWYH?

WYGIWYH simplifies money management by using a principle-first approach:

> Use what you earn this month for this month. Savings are tracked but treated as untouchable for future months.

This approach avoids the complexity of budgeting while providing a clear picture of your finances. WYGIWYH was built to address the lack of existing tools that meet these specific needs, offering features like multi-currency support, custom transactions, and API integration.

## Demo

Try out WYGIWYH at [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/) with the credentials:

> [!NOTE]
> E-mail: `demo@demo.com`
> 
> Password: `wygiwyhdemo`

Please note that the demo data is reset daily and most automation features are disabled.

## How to Use

WYGIWYH is designed to run using Docker and Docker Compose.

**Prerequisites:**

*   [Docker](https://docs.docker.com/engine/install/)
*   [docker-compose](https://docs.docker.com/compose/install/)

**Installation Steps:**

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

For local development, modify your `.env` file:
1.  Remove `URL`
2.  Set `HTTPS_ENABLED` to `false`
3.  Leave the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1])

You can now access the app at `localhost:OUTBOUND_PORT`.
For detailed instructions and environment variable configurations, please see the original [README](https://github.com/eitchtee/WYGIWYH).

### Latest Changes

Features are added to `main` when ready; to use the newest version, build from source or use the `:nightly` Docker tag.  See the [Dockerfiles](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod) for build instructions.

## Unraid

WYGIWYH is available on the Unraid Store.

To create the first user, open the container's console, then type `python manage.py createsuperuser`, you'll then be prompted to input your e-mail and password.

For detailed instructions, please see the original [README](https://github.com/eitchtee/WYGIWYH).

## Environment Variables

*   **DJANGO_ALLOWED_HOSTS:**  (string) A list of space-separated domains/IPs representing the hostnames WYGIWYH can serve.
*   **HTTPS_ENABLED:** (true|false) Enables secure cookies.
*   **URL:** (string) A list of space-separated domains/IPs (with protocol) representing trusted origins for unsafe requests.
*   **SECRET_KEY:** (string) Used for cryptographic signing, must be unique.
*   **DEBUG:** (true|false)  Enables debug mode.
*   **SQL_DATABASE:** (string) Name of your PostgreSQL database.
*   **SQL_USER:** (string) Username for your PostgreSQL database.
*   **SQL_PASSWORD:** (string) Password for your PostgreSQL database.
*   **SQL_HOST:** (string) Address for your PostgreSQL database.
*   **SQL_PORT:** (string) Port for your PostgreSQL database.
*   **SESSION_EXPIRY_TIME:** (int) The age of session cookies, in seconds.
*   **ENABLE_SOFT_DELETE:** (true|false)  Enables transaction soft delete.
*   **KEEP_DELETED_TRANSACTIONS_FOR:** (int) Time in days to keep soft deleted transactions.
*   **TASK_WORKERS:** (int)  Number of workers for async tasks.
*   **DEMO:** (true|false) Enables demo mode.
*   **ADMIN_EMAIL:** (string) Creates an admin account with this email.
*   **ADMIN_PASSWORD:** (string)  Creates an admin account with this password.
*   **CHECK_FOR_UPDATES:** (bool) Check and notify about new versions.

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) authentication via `django-allauth`. Set the following environment variables:

*   `OIDC_CLIENT_NAME`
*   `OIDC_CLIENT_ID`
*   `OIDC_CLIENT_SECRET`
*   `OIDC_SERVER_URL`
*   `OIDC_ALLOW_SIGNUP`

**Callback URL:** `https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

## How It Works

For more in-depth information, please refer to the project's [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Help Us Translate WYGIWYH!

Contribute to translations through [this link](https://translations.herculino.com/engage/wygiwyh/).  Login with your GitHub account.

## Caveats and Warnings

*   Not an accountant. Open an issue if you find inaccuracies.
*   Most calculations are done at runtime, potentially affecting performance.
*   Not a budgeting or double-entry accounting application.  Open a discussion if these features are desired.

## Built With

WYGIWYH is built with open-source tools, including:

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