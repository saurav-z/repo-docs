<!-- Django Ledger Logo -->
![django ledger logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)

# Django Ledger: A Powerful Django Accounting Engine

**Simplify complex financial tasks with Django Ledger, a robust, open-source accounting engine built for Django.**

[View the Original Repo](https://github.com/arrobalytics/django-ledger)

## Key Features

*   ✅ **Double-Entry Accounting:** Ensures accuracy and transparency in your financial records.
*   ✅ **Hierarchical Chart of Accounts:** Organize your finances with a flexible and customizable chart.
*   ✅ **Comprehensive Financial Statements:** Generate key reports including Income Statements, Balance Sheets, and Cash Flow Statements.
*   ✅ **Transaction Management:** Supports Ledgers, Journal Entries, and Transactions for detailed tracking.
*   ✅ **Financial Document Handling:** Includes Purchase Orders, Sales Orders, Bills, and Invoices.
*   ✅ **Advanced Functionality:** Offers financial ratio calculations, inventory management, and unit of measure support.
*   ✅ **Multi-Tenancy Support:** Ideal for applications managing multiple entities.
*   ✅ **Data Import:** Integrates with OFX & QFX file formats for easy data import.
*   ✅ **Seamless Integration:** Django Admin integration and a built-in Entity Management UI streamline operations.

## Getting Started

### Installation

Django Ledger is designed to integrate smoothly with your existing Django project.

1.  **Project Setup:** Ensure you have a working Django project. If you're new to Django, start with the [official Django tutorial](https://docs.djangoproject.com/en/4.2/intro/tutorial01/#creating-a-project).
2.  **Installation Options:**
    *   **Starter Template:** The quickest way to get started is with the zero-config Django Ledger starter template:  [django-ledger-starter](https://github.com/arrobalytics/django-ledger-starter).
    *   **Manual Installation:** Follow these steps to add Django Ledger to an existing project.

### Adding Django Ledger to your project:

1.  **Add to `INSTALLED_APPS`:**

```python
INSTALLED_APPS = [
    ...,
    'django_ledger',
    ...,
]
```

2.  **Include Context Preprocessor:**

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

3.  **Run Migrations:**

```bash
python manage.py migrate
```

4.  **Add URLs:** Add Django Ledger's URLs to your project's `urls.py`:

```python
from django.urls import include, path

urlpatterns = [
    ...,
    path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
    ...,
]
```

5.  **Run the Server:**

```bash
python manage.py runserver
```

6.  **Access Django Ledger:** Navigate to your project's ledger URL (usually http://127.0.0.1:8000/ledger) and log in with your superuser credentials.

## Deprecated Behavior Setting (v0.8.0+)

Starting with version v0.8.0, the `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting controls access to deprecated features and legacy behaviors.

*   **Default:** `False` (deprecated features are disabled)
*   To temporarily use deprecated features, set this to `True` in your Django settings while transitioning.

## Development Setup

Here's how to set up a development environment:

1.  **Clone the Repository:**

```bash
git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
```

2.  **Install PipEnv (if not already installed):**

```bash
pip install -U pipenv
```

3.  **Create and Activate Virtual Environment:**

```bash
pipenv install
pipenv shell
```

4.  **Apply Migrations:**

```bash
python manage.py migrate
```

5.  **Create a Superuser:**

```bash
python manage.py createsuperuser
```

6.  **Run the Development Server:**

```bash
python manage.py runserver
```

## Development Setup using Docker

1.  **Give executable permissions to entrypoint.sh**

```bash
sudo chmod +x entrypoint.sh
```

2.  **Add host '0.0.0.0' into ALLOWED_HOSTS in settings.py.**
3.  **Build and Run Docker Compose:**

```bash
docker compose up --build
```

4.  **Create a Superuser (in a separate terminal):**

```bash
docker ps
docker exec -it <container_id> /bin/sh
python manage.py createsuperuser
```

5.  **Access the Application:** Browse to http://0.0.0.0:8000/

## Testing

Run the test suite after setting up your development environment.

```bash
python manage.py test django_ledger
```

## Screenshots

<!-- Screenshots - Add alt text and captions where possible -->
![Django Ledger Entity Dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
![Django Ledger Balance Sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
![Django Ledger Income Statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
![Django Ledger Bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
![Django Ledger Invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statement Reports

![Balance Sheet Report](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
![Income Statement Report](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
![Cash Flow Statement Report](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)

## Get Involved

*   **Feature Requests/Bug Reports:** [Open an issue](https://github.com/arrobalytics/django-ledger/issues)
*   **Customization/Consulting:** [Contact Us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com
*   **Contribute:** See our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md)

**We welcome contributions from Python and Django developers with finance and accounting experience!**