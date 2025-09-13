[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: Powerful Accounting for Django

**Django Ledger** is a robust, open-source accounting engine that seamlessly integrates double-entry accounting into your Django applications.

[Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features of Django Ledger:

*   **Double-Entry Accounting:** Ensures accuracy and a complete audit trail.
*   **Hierarchical Chart of Accounts:** Organize your financial data logically.
*   **Financial Statements:** Generate key reports like Income Statements, Balance Sheets, and Cash Flow statements.
*   **Transaction Management:** Supports Ledgers, Journal Entries & Transactions.
*   **Order and Invoice Processing:** Handle Purchase Orders, Sales Orders, Bills, and Invoices efficiently.
*   **Financial Calculations:** Includes financial ratio calculations for in-depth analysis.
*   **Multi-Tenancy Support:** Design accounting for businesses with multiple entities.
*   **Data Import:** Import financial data via OFX & QFX files.
*   **Inventory Management:** Track inventory and associated costs.
*   **Unit of Measures:** Manage units for products and services.
*   **Bank Account Management:** Integrate bank account information for reconciliation.
*   **Django Admin Integration:** Leverages the Django admin for easy data management.
*   **Built-in Entity Management UI:** Manage multiple businesses and organizations.

## Getting Involved & Contributing

Django Ledger welcomes contributions! Help build a powerful accounting engine by submitting pull requests for bug fixes, enhancements, and new features.

*   **Report Bugs or Request Features:** Open an issue in the [repository](https://github.com/arrobalytics/django-ledger).
*   **Customization, Consulting & Advanced Features:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com.
*   **Contribute:** See our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md) for detailed instructions.

## Who Should Contribute?

We encourage contributions from developers with skills in:

*   Python and Django programming.
*   Accounting and financial expertise.
*   Developing robust accounting APIs.

## Installation

Django Ledger is a Django application. Ensure you have a working Django project and knowledge of Django fundamentals before installation. A great starting point is the [Django tutorial](https://docs.djangoproject.com/en/4.2/intro/tutorial01/#creating-a-project).

The easiest way to get started is to use the zero-config Django Ledger starter template, which you can find [here](https://github.com/arrobalytics/django-ledger-starter).

Otherwise, follow these steps to install it in your existing project:

### Add django_ledger to INSTALLED_APPS

```python
INSTALLED_APPS = [
    ...,
    'django_ledger',
    ...,
]
```

### Add Django Ledger Context Preprocessor

```python
TEMPLATES = [
    {
        'OPTIONS': {
            'context_processors': [
                '...',
                'django_ledger.context.django_ledger_context'  # Add this line to a context_processors list..
            ],
        },
    },
]
```

### Perform database migrations

```shell
python manage.py migrate
```

### Add URLs to your project's urls.py:

```python
from django.urls import include, path

urlpatterns = [
    ...,
    path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
    ...,
]
```

### Run your project

```shell
python manage.py runserver
```

*   Navigate to the Django Ledger root view (typically `http://127.0.0.1:8000/ledger` if you followed the installation steps).
*   Use your superuser credentials to log in.

## Deprecated Behavior Setting (v0.8.0+)

Since version v0.8.0, Django Ledger provides the `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting to control access to deprecated features:

*   **Default:** `False` (deprecated features are disabled)
*   To temporarily use deprecated features during migration, set it to `True` in your Django settings.

## Setting Up Django Ledger for Development

Django Ledger includes a basic development environment under the `__dev_env__/` folder (not for production):

1.  Navigate to your project directory.
2.  Clone the repository:

    ```bash
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```
3.  Install PipEnv (if you haven't already):

    ```bash
    pip install -U pipenv
    ```
4.  Create a virtual environment (specifying the Python interpreter if needed):

    ```bash
    pipenv install
    # or
    pipenv install --python PATH_TO_INTERPRETER
    ```
5.  Activate the environment:

    ```bash
    pipenv shell
    ```
6.  Apply migrations:

    ```bash
    python manage.py migrate
    ```
7.  Create a development Django superuser:

    ```bash
    python manage.py createsuperuser
    ```
8.  Run the development server:

    ```bash
    python manage.py runserver
    ```

## Setting Up Django Ledger for Development Using Docker

1.  Navigate to your project directory.
2.  Give execution permissions to entrypoint.sh:

    ```bash
    sudo chmod +x entrypoint.sh
    ```
3.  Add the host '0.0.0.0' to `ALLOWED_HOSTS` in `settings.py`.
4.  Build and run the Docker container:

    ```bash
    docker compose up --build
    ```
5.  Create a Django superuser (in a separate terminal):

    ```bash
    docker ps
    ```
    Find the container ID, then run:

    ```bash
    docker exec -it <containerId> /bin/sh
    python manage.py createsuperuser
    ```
6.  Access the application at `http://0.0.0.0:8000/` in your browser.

## Run Test Suite

After setting up your development environment, run the tests:

```bash
python manage.py test django_ledger
```

## Screenshots

These screenshots provide a visual overview of the application's functionality:

![django ledger entity dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
![django ledger balance sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
![django ledger income statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
![django ledger bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
![django ledger invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statements Screenshots

![balance_sheet_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
![income_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
![cash_flow_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)