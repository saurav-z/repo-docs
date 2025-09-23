[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: Powerful Double-Entry Accounting for Django

**Django Ledger** empowers developers to build robust financial applications with its comprehensive accounting engine, built for the Django framework.

[Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features of Django Ledger:

*   **Double-Entry Accounting:** Ensures accurate financial tracking.
*   **Hierarchical Chart of Accounts:** Organize financial data effectively.
*   **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow statements.
*   **Financial Transactions:** Ledger, Journal Entries & Transactions.
*   **Order Management:** Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Financial Analysis:** Built-in financial ratio calculations.
*   **Multi-tenancy Support:** Manage multiple businesses or entities within a single application.
*   **Import/Export:** OFX & QFX file import capabilities.
*   **Inventory Management:** Track and manage inventory levels.
*   **Unit of Measures:** Define and use units of measure for inventory items.
*   **Bank Account Management:** Store and manage bank account information.
*   **Django Admin Integration:** Seamless integration with the Django Admin interface.
*   **Built-in Entity Management UI:** Manage financial entities.

## Getting Involved & Contributing

We welcome contributions to Django Ledger!  Whether you're fixing bugs, enhancing existing features, or adding new functionality, your input is valuable.

*   **Feature Requests/Bug Reports:** Open an issue in the [GitHub Repository](https://github.com/arrobalytics/django-ledger).
*   **Software Customization, Advanced Features, Consulting:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com
*   **Contribution Guidelines:** See our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md).

### Who Should Contribute?

We're looking for contributors with skills in:

*   Python and Django programming
*   Finance and accounting principles
*   Building robust accounting APIs

## Installation

Django Ledger is a Django application.  You'll need a working Django project before installing.

The easiest way to start is to use the zero-config Django Ledger starter template ([django-ledger-starter](https://github.com/arrobalytics/django-ledger-starter)).

### Add to an Existing Project:

1.  **Add `django_ledger` to `INSTALLED_APPS`:**

```python
INSTALLED_APPS = [
    ...,
    'django_ledger',
    ...,
]
```

2.  **Add Django Ledger Context Preprocessor:**

```python
TEMPLATES = [
    {
        'OPTIONS': {
            'context_processors': [
                '...',
                'django_ledger.context.django_ledger_context'  # Add this line
            ],
        },
    },
]
```

3.  **Run Database Migrations:**

```shell
python manage.py migrate
```

4.  **Add URLs to `urls.py`:**

```python
from django.urls import include, path

urlpatterns = [
    ...,
    path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
    ...,
]
```

5.  **Run Your Project:**

```shell
python manage.py runserver
```

6.  **Access Django Ledger:**  Navigate to the URL assigned in your project's `urls.py` (usually `http://127.0.0.1:8000/ledger`) and log in with your superuser credentials.

## Deprecated Behavior (v0.8.0+)

The `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting controls access to deprecated features (default: `False`). Set this to `True` in your Django settings to temporarily use deprecated functionality.

## Setting up for Development

Django Ledger includes a development environment.

1.  Clone the repo:

```shell
git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
```

2.  Install PipEnv:

```shell
pip install -U pipenv
```

3.  Create and activate virtual environment:

```shell
pipenv install && pipenv shell
```

4.  Apply migrations:

```shell
python manage.py migrate
```

5.  Create a superuser:

```shell
python manage.py createsuperuser
```

6.  Run the development server:

```shell
python manage.py runserver
```

## Setting up for Development using Docker

1.  Navigate to your project directory.
2.  Give executable permissions to `entrypoint.sh`:

```shell
sudo chmod +x entrypoint.sh
```

3.  Add host '0.0.0.0' into ALLOWED_HOSTS in settings.py.
4.  Build and run the container:

```shell
docker compose up --build
```

5.  Create a Django superuser (in a separate terminal):

```shell
docker ps
docker exec -it <containerId> /bin/sh
python manage.py createsuperuser
```

6.  Access the application at `http://0.0.0.0:8000/`.

## Run Test Suite

```shell
python manage.py test django_ledger
```

## Screenshots

[Include all screenshots, which will help with SEO]