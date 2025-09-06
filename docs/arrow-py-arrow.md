# Arrow: Effortless Date and Time Manipulation in Python

**Simplify your Python date and time operations with Arrow, a user-friendly library that makes working with dates, times, and timestamps a breeze.**  Explore the original repository on [GitHub](https://github.com/arrow-py/arrow).

## Key Features

*   **Intuitive DateTime Replacement:**  A fully-implemented, drop-in replacement for Python's built-in `datetime` module.
*   **Python 3.8+ Compatibility:** Supports modern Python versions.
*   **Timezone Awareness:** Timezone-aware and UTC by default for accurate time handling.
*   **Simplified Creation:** Easy-to-use methods for creating dates and times from various inputs.
*   **Flexible Time Shifting:** The `shift` method supports relative offsets, including weeks.
*   **Automatic Formatting & Parsing:** Automatically formats and parses strings for ease of use.
*   **ISO 8601 Support:** Comprehensive support for the ISO 8601 standard.
*   **Timezone Conversion:** Seamlessly convert between timezones.
*   **Integration with Existing Libraries:** Supports `dateutil`, `pytz`, and `ZoneInfo` tzinfo objects.
*   **Time Span Generation:** Generates time spans, ranges, floors, and ceilings.
*   **Human-Friendly Dates:** Humanize dates and times with locale support.
*   **Extensible:** Extend Arrow for custom types.
*   **Type Hinting:** Full support for PEP 484-style type hints.

## Why Use Arrow?

Arrow addresses the usability shortcomings of Python's built-in datetime modules:

*   Reduces the need to import multiple modules (datetime, time, calendar, dateutil, etc.)
*   Simplifies working with multiple types (date, time, datetime, timedelta, etc.)
*   Streamlines timezone and timestamp conversions.
*   Prioritizes timezone awareness by default.
*   Adds functionality like ISO 8601 parsing, timespans, and humanization.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from an ISO 8601 string
arrow.get('2013-05-11T21:23:58.970460+07:00')  # <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()  # <Arrow [2024-10-27T10:30:00.000000+00:00]>

# Shift the time
utc = utc.shift(hours=-1)  # <Arrow [2024-10-27T09:30:00.000000+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')  # <Arrow [2024-10-27T02:30:00.000000-07:00]>

# Get the timestamp
local.timestamp()  # 1708195800.0

# Format the date/time
local.format()  # '2024-10-27 02:30:00 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')  # '2024-10-27 02:30:00 -07:00'

# Humanize the time
local.humanize()  # '7 hours ago'
local.humanize(locale='ko-kr')  # '7시간 전'
```

## Documentation

For detailed information, visit the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Start by:

1.  Reviewing existing [issues](https://github.com/arrow-py/arrow/issues).
2.  Forking the repository.
3.  Making your changes, and adding relevant tests.
4.  Run tests and linting (`tox && tox -e lint,docs` or `make build39 && make test && make lint`).
5.  Submit a pull request.

## Support Arrow

Support the project through [Open Collective](https://opencollective.com/arrow).