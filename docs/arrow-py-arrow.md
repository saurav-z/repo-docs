# Arrow: Python's Human-Friendly Date and Time Library

**Simplify date and time manipulation in Python with Arrow, a library designed for intuitive handling of dates, times, and timestamps.**  [Explore the Arrow repository on GitHub](https://github.com/arrow-py/arrow)

## Key Features of Arrow:

*   **User-Friendly API:**  Easily create, manipulate, and format dates and times with less code.
*   **Drop-in Replacement:** Fully implements and enhances the Python `datetime` type.
*   **Timezone Awareness:** Built-in support for timezones and UTC by default.
*   **Intuitive Creation:**  Simple methods for creating Arrow objects from various input formats.
*   **Flexible Manipulation:**  Utilize the `shift` method for relative offsets, including weeks.
*   **Automatic Formatting & Parsing:**  Intelligent handling of string formats.
*   **ISO 8601 Support:** Comprehensive support for the ISO 8601 standard.
*   **Timezone Conversion:** Seamless conversion between timezones.
*   **Integration:** Supports `dateutil`, `pytz`, and `ZoneInfo` tzinfo objects.
*   **Time Span Generation:**  Create time spans, ranges, floors, and ceilings.
*   **Humanization:**  Human-readable date and time representation, with growing locale support.
*   **Extensible:** Designed for creating your own Arrow-derived types.
*   **Type Hints:** Full support for PEP 484-style type hints.

## Why Use Arrow?

Arrow addresses the usability shortcomings of Python's built-in `datetime` and related modules:

*   **Reduces Complexity:** Avoids the need to import multiple modules like `datetime`, `time`, `calendar`, `dateutil`, and `pytz`.
*   **Simplifies Types:** Works with fewer types, like `date`, `time`, `datetime`, `tzinfo`, `timedelta`, and `relativedelta`.
*   **Improves Timezone Handling:** Makes timezone conversions and timestamp management more straightforward.
*   **Enhances Functionality:** Provides features like ISO 8601 parsing, timespans, and humanization.

## Quick Start

### Installation

Install Arrow using pip:

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

# Shift the time by a certain amount
utc = utc.shift(hours=-1)
# <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get a timestamp
local.timestamp()
# 1368303838.970460

# Format the time
local.format()
# '2013-05-11 13:23:58 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2013-05-11 13:23:58 -07:00'

# Humanize the time
local.humanize()
# 'an hour ago'
local.humanize(locale='ko-kr')
# '한시간 전'
```

## Documentation

For complete documentation, please visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

We welcome contributions!  Check out the [issue tracker](https://github.com/arrow-py/arrow/issues) for tasks, including those tagged "good first issue".  See the original README for instructions on contributing.

## Support Arrow

Consider supporting the project through [Open Collective](https://opencollective.com/arrow) for one-time or recurring donations.