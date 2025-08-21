# Arrow: Intuitive Date and Time Handling for Python

**Simplify date and time manipulation in Python with Arrow, a user-friendly library that makes working with dates and times a breeze.**  [View the project on GitHub](https://github.com/arrow-py/arrow)

## Key Features

*   **User-Friendly:** Provides a clean and intuitive API for all date and time operations.
*   **Drop-in Replacement:** Fully implements and extends the Python `datetime` type.
*   **Timezone Awareness:** Handles timezones gracefully, with UTC as the default.
*   **Easy Creation:** Simple methods for creating dates and times from various input formats.
*   **Flexible Manipulation:** Offers `shift` methods for relative offsets (including weeks), as well as time spans, ranges, and floors/ceilings.
*   **Automatic Formatting and Parsing:**  Formats and parses strings automatically, with robust ISO 8601 support.
*   **Timezone Conversion:** Seamlessly converts between timezones.
*   **Humanization:** Humanizes dates and times with localization support (multiple locales available).
*   **Type Hinting:**  Full support for PEP 484-style type hints.

## Why Choose Arrow?

Arrow addresses the common pain points of working with Python's built-in datetime modules:

*   **Reduces complexity:** Avoids the need to juggle multiple modules and types (datetime, time, tzinfo, etc.).
*   **Simplifies Timezone Handling:** Makes timezone conversions and timestamp management straightforward.
*   **Enhances Usability:** Offers features such as ISO 8601 parsing, time spans, and humanization.

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
arrow.get('2013-05-11T21:23:58.970460+07:00')
# <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time by a given number of hours
utc = utc.shift(hours=-1)
# <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a local timezone
local = utc.to('US/Pacific')
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
local.timestamp()
# 1368303838.970460

# Format the date and time
local.format()
# '2013-05-11 13:23:58 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2013-05-11 13:23:58 -07:00'

# Humanize the date and time
local.humanize()
# 'an hour ago'
local.humanize(locale='ko-kr')
# '한시간 전'
```

## Documentation

For detailed information, see the full documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

We welcome contributions!  See the [contributing guidelines](https://github.com/arrow-py/arrow/blob/master/CONTRIBUTING.md) for instructions on how to get involved.

## Support Arrow

Support Arrow by donating via [Open Collective](https://opencollective.com/arrow).