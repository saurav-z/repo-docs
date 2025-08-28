# Arrow: Human-Friendly Dates and Times for Python

**Simplify your Python date and time manipulations with Arrow, a library designed to make working with dates, times, and timestamps a breeze.**  Learn more on the [original repository](https://github.com/arrow-py/arrow).

## Key Features

*   **Intuitive datetime replacement:** Fully implements and extends Python's `datetime` for easier use.
*   **Python 3.8+ support:** Compatible with the latest Python versions.
*   **Timezone-aware by default:**  Works with UTC by default, simplifying timezone conversions.
*   **Simplified creation:** Easily create dates and times from various input formats.
*   **Flexible time shifting:** Use the `shift` method for relative offsets (including weeks).
*   **Automatic formatting and parsing:**  Handles string formats with ease.
*   **ISO 8601 support:**  Comprehensive support for the ISO 8601 standard.
*   **Timezone conversion:** Seamlessly convert between timezones.
*   **dateutil, pytz, and ZoneInfo support:**  Compatible with common timezone libraries.
*   **Time spans and ranges:** Generate time spans, ranges, floors, and ceilings.
*   **Human-readable formatting:**  Humanize dates and times in various locales.
*   **Extensible:** Easily create custom Arrow-derived types.
*   **Type hints:**  Full support for PEP 484-style type hints.

## Why Arrow?

Arrow solves the usability problems of Python's built-in date and time modules by:

*   Reducing the number of modules you need to import.
*   Simplifying the various date and time types.
*   Making timezone conversions and timestamp handling straightforward.
*   Offering more intuitive methods for common tasks.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from a string
a = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(a)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time
utc = utc.shift(hours=-1)
print(utc)  # Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
print(local)  # Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
print(local.timestamp())  # Output: 1368303838.97046

# Format the time
print(local.format())  # Output: 2013-05-11 13:23:58 -07:00
print(local.format('YYYY-MM-DD HH:mm:ss ZZ'))  # Output: 2013-05-11 13:23:58 -07:00

# Humanize the time
print(local.humanize())  # Output: an hour ago
print(local.humanize(locale='ko-kr'))  # Output: 한시간 전
```

## Documentation

For detailed information, visit the official documentation at `arrow.readthedocs.io <https://arrow.readthedocs.io>`_.

## Contributing

Contributions are welcome!  Please review the [contributing guidelines](https://github.com/arrow-py/arrow/blob/master/CONTRIBUTING.md) for details on how to submit code, documentation, or locale updates.

## Support Arrow

Support the project by donating on the [Open Collective](https://opencollective.com/arrow).