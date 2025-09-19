<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker
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
  <a href="#built-with">Built with</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">Original Repository</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is your key to straightforward financial management, offering a robust, no-budget approach.  This open-source finance tracker helps you track your income, expenses, and investments, and is designed for simplicity and flexibility.

[Screenshots of Key Features - Monthly, Yearly, Net Worth, Calendar, Transactions]

## Why WYGIWYH?

Traditional budgeting can be complex, and WYGIWYH simplifies your finances with a powerful set of features. WYGIWYH is based on a simple principle:

> Use what you earn this month for this month. Any savings are tracked but treated as untouchable for future months.

This approach simplifies money management while providing a clear view of your financial health.  The project was born out of the need for a finance tracker that offered multi-currency support, no budgeting constraints, web app usability, API support, and custom transaction rules.  WYGIWYH delivers on these needs.

## Key Features

WYGIWYH provides a comprehensive suite of features for effortless finance tracking:

*   **Unified Transaction Tracking:**  Organize all income and expenses in one place.
*   **Multi-Account Support:**  Track funds across various accounts (banks, wallets, investments).
*   **Multi-Currency Support:** Manage transactions and balances in multiple currencies dynamically.
*   **Custom Currency Creation:**  Define custom currencies for crypto, rewards, or other models.
*   **Automated Transaction Rules:** Automatically adjust transactions with customizable rules.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Monitor recurring investments, including crypto and stocks.
*   **API Support:** Integrate with existing services for automated transaction synchronization.

## Demo

Experience WYGIWYH firsthand on our demo site: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

>   **Demo Credentials:**
>   *   Email: `demo@demo.com`
>   *   Password: `wygiwyhdemo`
>
>   *Please note:  Demo data is wiped daily.*
>   *Automation features (API, Rules, Exchange Rates, Import/Export) are disabled in the demo.*

## How To Use

WYGIWYH uses Docker for easy setup and deployment.  Here's how to get started:

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

>   **Unraid Users:** Utilize the Unraid template, available in the [Unraid section](#unraid), which can be found below.
>   **Important:**  Consult the [Environment Variables](#environment-variables) section for configuration details.

### Running Locally

To run WYGIWYH locally:

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

Access the app via `localhost:OUTBOUND_PORT`.

>   **Note:**
>   *   For Tailscale or similar services, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
>   *   For non-localhost IPs, include the IP in `DJANGO_ALLOWED_HOSTS` (without `http://`).

### Latest Changes

For the most up-to-date features, use the `:nightly` tag in Docker.  Be aware of potential undocumented changes.

Find the Dockerfiles [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

## Unraid

[nwithan8](https://github.com/nwithan8) has provided a helpful Unraid template. Check out the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is available on the Unraid Store. You'll need your own PostgreSQL (version 15+) database.

Create the first user by accessing the container console (Unraid UI -> Docker -> WYGIWYH -> Console) and typing `python manage.py createsuperuser`.

## Environment Variables

Customize WYGIWYH with these environment variables:

| Variable                      | Type         | Default                           | Description                                                                                                                                                                                                                               |
| :---------------------------- | :----------- | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DJANGO_ALLOWED_HOSTS          | string       | localhost 127.0.0.1               |  A space-separated list of allowed hostnames and IPs. [Learn More](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts).                                                                                             |
| HTTPS_ENABLED                 | true/false   | false                             |  Enable secure cookies (HTTPS).                                                                                                                                                                                                           |
| URL                           | string       | http://localhost http://127.0.0.1 |  Trusted origins for unsafe requests. [Learn More](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins).                                                                                                        |
| SECRET_KEY                    | string       | ""                                |  A unique, unpredictable secret key.                                                                                                                                                                                                        |
| DEBUG                         | true/false   | false                             |  Enable debug mode (for development only).                                                                                                                                                                                                   |
| SQL_DATABASE                  | string       | None (required)                   |  Your PostgreSQL database name.                                                                                                                                                                                                           |
| SQL_USER                      | string       | user                              |  PostgreSQL username.                                                                                                                                                                                                                        |
| SQL_PASSWORD                  | string       | password                          |  PostgreSQL password.                                                                                                                                                                                                                        |
| SQL_HOST                      | string       | localhost                         |  PostgreSQL host address.                                                                                                                                                                                                                   |
| SQL_PORT                      | string       | 5432                              |  PostgreSQL port.                                                                                                                                                                                                                          |
| SESSION_EXPIRY_TIME           | integer      | 2678400 (31 days)                 |  Session cookie age (in seconds).                                                                                                                                                                                                         |
| ENABLE_SOFT_DELETE            | true/false   | false                             |  Enable soft deletes for transactions.                                                                                                                                                                                                       |
| KEEP_DELETED_TRANSACTIONS_FOR | integer      | 365                               |  Days to keep soft-deleted transactions (if `ENABLE_SOFT_DELETE` is true). 0 for indefinite.                                                                                                                                           |
| TASK_WORKERS                  | integer      | 1                                 |  Number of workers for asynchronous tasks.                                                                                                                                                                                                  |
| DEMO                          | true/false   | false                             |  Enable demo mode.                                                                                                                                                                                                                          |
| ADMIN_EMAIL                   | string       | None                              |  Automatically create an admin account with this email (requires `ADMIN_PASSWORD`).                                                                                                                                                     |
| ADMIN_PASSWORD                | string       | None                              |  Automatically create an admin account with this password (requires `ADMIN_EMAIL`).                                                                                                                                                     |
| CHECK_FOR_UPDATES             | bool         | true                              |  Check for and notify users about new versions (checks GitHub API every 12 hours).                                                                                                                                                       |

## OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) for user authentication.

>   **Note:**  Currently, only OpenID Connect is supported. Please create an issue if you need something else.

Set these environment variables for OIDC:

| Variable             | Description                                                                                                                                                                                                                                            |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OIDC_CLIENT_NAME`   | Provider name (displayed in the login page). Defaults to `OpenID Connect`.                                                                                                                                                                               |
| `OIDC_CLIENT_ID`     | Your OIDC provider's Client ID.                                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | Your OIDC provider's Client Secret.                                                                                                                                                                                                                       |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider (e.g., `https://your-provider.com/auth/realms/your-realm`).  `django-allauth` uses this to discover endpoints.                                                                                                     |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic creation of accounts after successful authentication. Defaults to `true`.                                                                                                                                                                 |

**Callback URL (Redirect URI):**

When configuring your OIDC provider, use this callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH instance URL, and `<OIDC_CLIENT_NAME>` with the slugified `OIDC_CLIENT_NAME` value, or `openid-connect` if not set.

## How It Works

Explore our [Wiki](https://github.com/eitchtee/WYGIWYH/wiki) for detailed information.

## Help Us Translate WYGIWYH!

Contribute to WYGIWYH's translation efforts:

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

>   **Note:**  Log in with your GitHub account.

## Caveats and Warnings

-   I'm not an accountant; some terms or calculations might be incorrect.  Please open an issue if you spot any improvements.
-   Calculations occur at runtime, potentially impacting performance.
-   WYGIWYH is *not* a budgeting or double-entry accounting app. If you need those features, consider other options, or open a discussion if you want them included in WYGIWYH.

## Built With

WYGIWYH is powered by these amazing open-source tools:

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