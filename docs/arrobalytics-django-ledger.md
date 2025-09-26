![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)

# Django Ledger: Your Powerful Double-Entry Accounting Engine for Django

**Django Ledger empowers developers to build robust financial applications with ease, offering a comprehensive and streamlined API.**

[View the original repository on GitHub](https://github.com/arrobalytics/django-ledger)

[FREE Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features

*   **Double-Entry Accounting:** Ensures accurate financial tracking.
*   **Hierarchical Chart of Accounts:** Organize your finances efficiently.
*   **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow statements.
*   **Transaction Management:** Ledgers, Journal Entries & Transactions for detailed record-keeping.
*   **Order & Invoice Management:** Handle Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Financial Ratios:** Calculate key financial ratios for analysis.
*   **Multi-tenancy Support:** Manage finances for multiple entities within a single application.
*   **Import & Export:** OFX & QFX file import for easy data migration.
*   **Inventory & Unit of Measures:** Streamline inventory control.
*   **Django Admin Integration:** Seamless integration with the Django Admin interface.
*   **Built-in Entity Management UI:** Ready-to-use UI for managing financial entities.
*   **Closing Entries:** Supports creating closing entries for accounting periods.
*   **Bank Account Information:** Store and manage bank account details.

## Getting Involved

We welcome contributions to Django Ledger! Whether you're a seasoned Python/Django developer or have accounting expertise, your help is valuable.

*   **Feature Requests/Bug Reports:** Open an issue in the [repository](https://github.com/arrobalytics/django-ledger).
*   **For custom development, consulting, or advanced features:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com
*   **Contribute:** See our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md).

## Who Should Contribute?

We're looking for contributors with experience in:

*   Python and Django programming
*   Finance and accounting principles
*   Building robust accounting APIs

## Installation

Django Ledger is a Django application. Familiarity with Django and a working Django project is required. A good place to start is [here](https://docs.djangoproject.com/en/4.2/intro/tutorial01/#creating-a-project).

**Recommended:** Use the zero-config [Django Ledger starter template](https://github.com/arrobalytics/django-ledger-starter) for the easiest setup.

**Or, add to an existing project:**

### 1. Add to `INSTALLED_APPS`

```python
INSTALLED_APPS = [
    ...,
    'django_ledger',
    ...,
]
```

### 2. Add Context Preprocessor

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

### 3. Perform Database Migrations

```shell
python manage.py migrate
```

### 4. Add URLs to `urls.py`

```python
from django.urls import include, path

urlpatterns = [
    ...,
    path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
    ...,
]
```

### 5. Run Your Project

```shell
python manage.py runserver
```

*   Navigate to the Django Ledger root view (e.g., `http://127.0.0.1:8000/ledger`).
*   Log in with your superuser credentials.

## Deprecated Behavior Setting (v0.8.0+)

The `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting controls access to deprecated features.  Default: `False`. Set to `True` to temporarily use deprecated features while transitioning.

## Setting Up Django Ledger for Development

Follow these steps to contribute:

1.  Navigate to your project directory.
2.  Clone the repository:

    ```shell
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```
3.  Install PipEnv (if needed):

    ```shell
    pip install -U pipenv
    ```
4.  Create and activate a virtual environment:

    ```shell
    pipenv install
    pipenv shell
    ```

    *   Or, specify Python version:

        ```shell
        pipenv install --python PATH_TO_INTERPRETER
        ```
5.  Apply migrations:

    ```shell
    python manage.py migrate
    ```
6.  Create a superuser:

    ```shell
    python manage.py createsuperuser
    ```
7.  Run the development server:

    ```shell
    python manage.py runserver
    ```

## How To Set Up Django Ledger for Development using Docker

1.  Navigate to your projects directory.
2.  Give executable permissions to `entrypoint.sh`:

    ```shell
    sudo chmod +x entrypoint.sh
    ```
3.  Add host '0.0.0.0' to `ALLOWED_HOSTS` in `settings.py`.
4.  Build and run the container:

    ```shell
    docker compose up --build
    ```
5.  Create a Django superuser:

    ```shell
    docker ps
    docker exec -it <containerId> /bin/sh
    python manage.py createsuperuser
    ```
6.  Access Django Ledger in your browser at `http://0.0.0.0:8000/`.

## Run Test Suite

```shell
python manage.py test django_ledger
```

## Screenshots

*(Image URLs provided in original README)*

*   [Entity Dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
*   [Balance Sheet](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
*   [Income Statement](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
*   [Cash Flow Statement](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)
*   [Bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
*   [Invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)