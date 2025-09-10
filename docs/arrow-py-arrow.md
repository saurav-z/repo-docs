# Arrow: Human-Friendly Dates and Times for Python

**Simplify your Python date and time manipulations with Arrow, a powerful library that makes working with dates, times, and timestamps a breeze.**  ([Original Repository](https://github.com/arrow-py/arrow))

Arrow offers a more intuitive and user-friendly approach to working with dates and times in Python, overcoming the complexities of the built-in datetime module.

## Key Features

*   **Drop-in Replacement:** Fully compatible with Python's `datetime` objects.
*   **Python Version Support:** Compatible with Python 3.8 and later.
*   **Timezone Awareness:** Defaults to timezone-aware and UTC for easier handling.
*   **Simplified Creation:** Easy-to-use methods for creating dates and times from various input formats.
*   **Flexible Manipulation:** Shift dates with relative offsets, including weeks, for powerful calculations.
*   **Automatic Formatting and Parsing:** Effortlessly format and parse strings with built-in capabilities.
*   **ISO 8601 Support:** Comprehensive support for the ISO 8601 standard.
*   **Timezone Conversion:** Seamlessly convert between timezones.
*   **Integration with Existing Libraries:** Supports `dateutil`, `pytz`, and `ZoneInfo` objects.
*   **Time Spans and Ranges:** Generate time spans, ranges, floors, and ceilings for microsecond to year intervals.
*   **Humanization:** Convert dates and times into human-readable formats, with support for multiple locales.
*   **Extensibility:** Easily create custom Arrow-derived types.
*   **Type Hinting:** Full support for PEP 484-style type hints.

## Why Use Arrow?

Compared to Python's built-in `datetime` and related modules, Arrow provides a more streamlined and user-friendly experience by addressing common pain points:

*   **Consolidated Modules:** Reduces the need to juggle multiple modules (datetime, time, calendar, dateutil, pytz, etc.).
*   **Simplified Types:** Works with fewer types (date, time, datetime, tzinfo, timedelta, relativedelta, etc.).
*   **Intuitive Timezone Handling:** Makes timezone conversions and timestamp manipulations much easier.
*   **Timezone Defaults:** Uses a sensible timezone-aware default.
*   **Enhanced Functionality:** Provides features like ISO 8601 parsing, timespans, and humanization.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Parse a datetime string
arrow.get('2013-05-11T21:23:58.970460+07:00')
# <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time back by one hour
utc = utc.shift(hours=-1)
# <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
local.timestamp()
# 1368303838.970460

# Format the datetime
local.format()
# '2013-05-11 13:23:58 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2013-05-11 13:23:58 -07:00'

# Humanize the datetime
local.humanize()
# 'an hour ago'
local.humanize(locale='ko-kr')
# '한시간 전'
```

## Documentation

Find detailed documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  See the original repository for detailed instructions on how to contribute code and localizations.

## Support Arrow

Consider supporting the project through the [Open Collective](https://opencollective.com/arrow) platform to help ensure its continued development.