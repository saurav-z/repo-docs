[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: Robust Double-Entry Accounting for Django

**Django Ledger** empowers Django developers with a comprehensive financial management system, simplifying complex accounting tasks with a user-friendly API.

**Key Features:**

*   ✅ **Double-Entry Accounting:**  Ensures accurate financial record-keeping.
*   🏦 **Hierarchical Chart of Accounts:** Organize your financial data effectively.
*   📊 **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow statements.
*   🧾 **Transaction Management:** Handle Purchase Orders, Sales Orders, Bills, and Invoices.
*   📈 **Financial Ratio Calculations:** Gain valuable financial insights.
*   🏢 **Multi-tenancy Support:**  Manage multiple businesses or entities within one system.
*   📒 **Ledgers, Journal Entries & Transactions:** Core accounting components.
*   📤 **OFX & QFX File Import:** Seamlessly import financial data.
*   🔄 **Closing Entries:** Automate year-end or period-end closing processes.
*   📦 **Inventory Management:** Track and manage your inventory.
*   📏 **Unit of Measures:**  Define and track different units of measurement.
*   🏦 **Bank Account Information:** Store and manage bank account details.
*   ⚙️ **Django Admin Integration:**  Easily manage data through the Django admin interface.
*   👥 **Built-in Entity Management UI:** User-friendly interface for managing entities.

## Getting Started

**Resources:**

*   [FREE Get Started Guide](https://www.djangoledger.com/get-started)
*   [Join our Discord](https://discord.gg/c7PZcbYgrc)
*   [Documentation](https://django-ledger.readthedocs.io/en/latest/)
*   [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Installation

Django Ledger is a Django application. This section outlines the necessary steps to integrate Django Ledger into your existing or new Django project.

**Prerequisites:**

*   Working knowledge of Django.
*   A working Django project.

**Steps:**

1.  **Add django\_ledger to INSTALLED\_APPS in your Django Project's `settings.py`:**

    ```python
    INSTALLED_APPS = [
        ...,
        'django_ledger',
        ...,
    ]
    ```

2.  **Add Django Ledger Context Preprocessor in your Django Project's `settings.py`:**

    ```python
    TEMPLATES = [
        {
            'OPTIONS': {
                'context_processors': [
                    '...',
                    'django_ledger.context.django_ledger_context'  # Add this line to a context_processors list.
                ],
            },
        },
    ]
    ```

3.  **Perform database migrations:**

    ```shell
    python manage.py migrate
    ```

4.  **Add URLs to your project's `urls.py`:**

    ```python
    from django.urls import include, path

    urlpatterns = [
        ...,
        path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
        ...,
    ]
    ```

5.  **Run your project:**

    ```shell
    python manage.py runserver
    ```

6.  **Access Django Ledger:** Navigate to the ledger root view in your project, which is typically `http://127.0.0.1:8000/ledger` (if you followed the above installation guide). Use your superuser credentials to login.

**Alternative: Zero-Config Starter Template**

For the easiest setup, consider using the zero-config Django Ledger starter template:  [django-ledger-starter](https://github.com/arrobalytics/django-ledger-starter).

## Deprecated Behavior (v0.8.0+)

*   Starting with v0.8.0, a setting `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` controls access to deprecated features.
    *   Default: `False` (deprecated features are disabled by default)
    *   To temporarily use deprecated features, set this to `True` in your Django settings.

## Development Environment Setup

This section guides you on setting up a development environment for contributing to the project.

1.  Navigate to your projects directory.
2.  Clone the repository:

    ```shell
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```

3.  Install PipEnv (if not already installed):

    ```shell
    pip install -U pipenv
    ```

4.  Create and activate the virtual environment:

    ```shell
    pipenv install
    pipenv shell
    ```

5.  Apply migrations:

    ```shell
    python manage.py migrate
    ```

6.  Create a Django superuser:

    ```shell
    python manage.py createsuperuser
    ```

7.  Run the development server:

    ```shell
    python manage.py runserver
    ```

## Docker Development Setup

1.  Navigate to your project directory.
2.  Give executable permissions to `entrypoint.sh`:

    ```shell
    sudo chmod +x entrypoint.sh
    ```

3.  Add host '0.0.0.0' to `ALLOWED_HOSTS` in `settings.py`.
4.  Build and run the container:

    ```shell
    docker compose up --build
    ```

5.  Create a Django superuser (in a separate terminal):

    ```shell
    docker ps
    docker exec -it <container_id> /bin/sh
    python manage.py createsuperuser
    ```

6.  Access the application in your browser at `http://0.0.0.0:8000/`.

## Running Tests

After setting up the development environment, run tests with:

```shell
python manage.py test django_ledger
```

## Contributing

We welcome contributions!  See our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md) for details.

*   **Feature Requests/Bug Reports:** Open an issue in the repository.
*   **Customization & Consulting:**  [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com.

### Who Should Contribute?

We are looking for contributors with:

*   Python and Django programming skills
*   Finance and accounting expertise
*   Interest in developing a robust accounting engine API

## Screenshots

*   ![Django Ledger Entity Dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
*   ![Django Ledger Balance Sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
*   ![Django Ledger Income Statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
*   ![Django Ledger Bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
*   ![Django Ledger Invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statements Screenshots

*   ![Balance Sheet](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
*   ![Income Statement](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
*   ![Cash Flow Statement](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)