<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker
  <br>
</h1>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#demo">Demo</a> •
  <a href="#help-us-translate">Help Us Translate</a> •
  <a href="#caveats-and-warnings">Caveats and Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH (What You Get Is What You Have)** is a finance tracker that simplifies money management with a no-budget, straightforward approach.

<img src=".github/img/monthly_view.png" width="18%"></img> <img src=".github/img/yearly.png" width="18%"></img> <img src=".github/img/networth.png" width="18%"></img> <img src=".github/img/calendar.png" width="18%"></img> <img src=".github/img/all_transactions.png" width="18%"></img>

## Key Features

*   **Unified Transaction Tracking:** Record all income and expenses in one place.
*   **Multi-Account Support:** Track money and assets across various accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:** Manage transactions and balances in different currencies with ease.
*   **Custom Currency Options:** Create custom currencies for crypto, rewards points, or any other models.
*   **Automated Transaction Rules:** Automate adjustments using customizable rules.
*   **Built-in Dollar-Cost Averaging (DCA) Tracker:** Track recurring investments.
*   **API Support:** Integrate with other services for automated transaction synchronization.

## Why WYGIWYH?

WYGIWYH simplifies money management by focusing on this principle:

> Use what you earn this month for this month. Any savings are tracked but treated as untouchable for future months.

This allows you to avoid dipping into savings while clearly seeing where your money goes.  WYGIWYH emerged from the need for a finance tracker that offered multi-currency support, no budgeting constraints, web app usability, API integration, and custom transaction rules.

## Demo

Try WYGIWYH with the demo credentials:

> [!NOTE]
> E-mail: `demo@demo.com`
> 
> Password: `wygiwyhdemo`

Keep in mind that **any data you add will be wiped in 24 hours or less**. And that **most automation features like the API, Rules, Automatic Exchange Rates and Import/Export are disabled**.

## How To Use

WYGIWYH uses Docker and Docker Compose for easy setup.

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

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).
4.  Access the application via `localhost:OUTBOUND_PORT`.

> [!NOTE]
> -   If using Tailscale or similar, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
> -   For non-localhost IPs, add them to `DJANGO_ALLOWED_HOSTS` without `http://`.

### Latest Changes

Features are added to `main` when ready. For the latest version, build from source or use the `:nightly` tag on Docker.  See the [Dockerfiles](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

### Unraid

WYGIWYH is available on the Unraid Store, with a template provided by [nwithan8](https://github.com/nwithan8/unraid_templates). You'll need to provision your own PostgreSQL database (version 15 or up). Open the container's console in the Unraid UI and run `python manage.py createsuperuser` to create the first user.

### Environment Variables

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                                                                 |
|-------------------------------|-------------|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DJANGO_ALLOWED_HOSTS          | string      | localhost 127.0.0.1               | Allowed host/domain names for the WYGIWYH site. [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts)                                                                                                          |
| HTTPS_ENABLED                 | true\|false | false                             | Enables secure cookies.                                                                                                                                                                                                                     |
| URL                           | string      | http://localhost http://127.0.0.1 | Trusted origins for unsafe requests (e.g., POST). [More details](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins)                                                                                              |
| SECRET_KEY                    | string      | ""                                | Cryptographic signing key.  Set this to a unique, unpredictable value.                                                                                                                                                                        |
| DEBUG                         | true\|false | false                             | Enables or disables DEBUG mode (for development). Don't use in production.                                                                                                                                                               |
| SQL_DATABASE                  | string      | None *required                    | PostgreSQL database name.                                                                                                                                                                                                                    |
| SQL_USER                      | string      | user                              | PostgreSQL username.                                                                                                                                                                                                                         |
| SQL_PASSWORD                  | string      | password                          | PostgreSQL password.                                                                                                                                                                                                                         |
| SQL_HOST                      | string      | localhost                         | PostgreSQL host address.                                                                                                                                                                                                                     |
| SQL_PORT                      | string      | 5432                              | PostgreSQL port.                                                                                                                                                                                                                           |
| SESSION_EXPIRY_TIME           | int         | 2678400 (31 days)                 | Session cookie age in seconds.                                                                                                                                                                                                            |
| ENABLE_SOFT_DELETE            | true\|false | false                             | Enables soft-deletion of transactions. Deleted transactions will remain in the database. Useful for imports and avoiding duplicate entries.                                                                                                |
| KEEP_DELETED_TRANSACTIONS_FOR | int         | 365                               | Time in days to keep soft deleted transactions for. If 0, will keep all transactions indefinitely. Only works if ENABLE_SOFT_DELETE is true.                                                                                                  |
| TASK_WORKERS                  | int         | 1                                 | Number of workers for async tasks.                                                                                                                                                                                                          |
| DEMO                          | true\|false | false                             | Enables demo mode.                                                                                                                                                                                                                           |
| ADMIN_EMAIL                   | string      | None                              | Automatically creates an admin account with this email. Requires `ADMIN_PASSWORD` to be set.                                                                                                                                               |
| ADMIN_PASSWORD                | string      | None                              | Automatically creates an admin account with this password. Requires `ADMIN_EMAIL` to be set.                                                                                                                                               |
| CHECK_FOR_UPDATES             | bool        | true                              | Checks for and notifies users about new versions.                                                                                                                                                                                        |

### OIDC Configuration

WYGIWYH supports OpenID Connect (OIDC) for authentication.

| Variable             | Description                                                                                                                                                                                                                                                            |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider, displayed on the login page. Defaults to `OpenID Connect`.                                                                                                                                                                                  |
| `OIDC_CLIENT_ID`     | The Client ID from your OIDC provider.                                                                                                                                                                                                                               |
| `OIDC_CLIENT_SECRET` | The Client Secret from your OIDC provider.                                                                                                                                                                                                                           |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` uses this for endpoints.                                                                            |
| `OIDC_ALLOW_SIGNUP`  | Allow automatic creation of accounts on successful authentication. Defaults to `true`.                                                                                                                                                                              |

**Callback URL:**

When configuring your OIDC provider, the default callback URL is:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH instance's URL and `<OIDC_CLIENT_NAME>` with either the value of the `OIDC_CLIENT_NAME` variable or `openid-connect` if you haven't defined it.

## Help Us Translate

Help localize WYGIWYH by translating the interface!

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your github account

## Caveats and Warnings

*   I'm not an accountant; some terms or calculations may be incorrect. Please open an issue if you find any errors.
*   Most calculations are done at runtime, potentially leading to performance degradation.
*   This is not a budgeting or double-entry accounting application. If you need those features, consider alternative tools.

## Built with

WYGIWYH utilizes these open-source tools:

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
Key improvements and optimizations:

*   **SEO-Friendly Title:**  "WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker" is more keyword-rich.
*   **Concise Hook:** The first sentence immediately introduces the product and its core value.
*   **Clear Headings:**  Uses consistent, descriptive headings for better organization and readability.
*   **Bulleted Key Features:** Easy-to-scan and emphasizes the core functionality.
*   **Concise Explanations:**  Simplified language and removed unnecessary details.
*   **Call to Action:** Encourages users to try the demo and help with translation.
*   **GitHub Link:** Added a prominent link back to the GitHub repository.
*   **Environment Variables Table:** Improved formatting and added detailed explanations.
*   **OIDC Configuration:** Improved description of the OIDC configuration options.
*   **Concise Summary:** Provided a quick summary.
*   **Consistent Formatting:** Used a consistent Markdown style.
*   **Unraid Section:** Improved Unraid section and highlighted the PostgreSQL requirement.