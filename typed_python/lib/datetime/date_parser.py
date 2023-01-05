# Copyright 2017-2020 typed_python Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typed_python import Class, Dict, Final
from typed_python import Entrypoint, ListOf
from typed_python.lib.datetime.chrono import Chrono
from typed_python.lib.datetime.date_time import (
    TimeZone,
    UTC,
    TimeZoneChecker,
    TimeOfDay,
    Date,
    DateTime,
    FixedOffsetTimezone,
)

JAN = "jan"
FEB = "feb"
MAR = "mar"
APR = "apr"
MAY = "may"
JUN = "jun"
JUL = "jul"
AUG = "aug"
SEP = "sep"
OCT = "oct"
NOV = "nov"
DEC = "dec"

JANUARY = "january"
FEBRUARY = "february"
MARCH = "march"
APRIL = "april"
JUNE = "june"
JULY = "july"
AUGUST = "august"
SEPT = "sept"
SEPTEMBER = "september"
OCTOBER = "october"
NOVEMBER = "november"
DECEMBER = "december"

T = "t"
PLUS = "+"
COLON = ":"
BACKSLASH = "/"
DASH = "-"
COMMA = ","
SPACE = " "
DOT = "."
AM = "am"
PM = "pm"

MONTH_TO_INT = Dict(str, int)(
    {
        JAN: 1,
        JANUARY: 1,
        FEB: 2,
        FEBRUARY: 2,
        MAR: 3,
        MARCH: 3,
        APR: 4,
        APRIL: 4,
        MAY: 5,
        JUN: 6,
        JUNE: 6,
        JUL: 7,
        JULY: 7,
        AUG: 8,
        AUGUST: 8,
        SEP: 9,
        SEPT: 9,
        SEPTEMBER: 9,
        OCT: 10,
        OCTOBER: 10,
        NOV: 11,
        NOVEMBER: 11,
        DEC: 12,
        DECEMBER: 12,
    }
)


@Entrypoint
def is_month(month_str: str) -> bool:
    """
    Tests if string represents a valid month
    Parameters:
        month_str: The month string (case insenstive). Examples: 'Jan', 'January'
    Returns:
        True if the month string is valid, False otherwise
    """
    return month_str in MONTH_TO_INT


@Entrypoint
def is4d(str: str) -> bool:
    """
    Tests if a string is 4 digits.
    Parameters:
        str(string):
    Returns:
        True if the input is a 4 digit string
    """
    return len(str) == 4 and str.isdigit()


@Entrypoint
def is2d(str: str) -> bool:
    """
    Tests if a string is 2 digits.
    Parameters:
        str(string):
    Returns:
        True if the input is a 2 digit string
    """
    return len(str) == 2 and str.isdigit()


class DateParser(Class, Final):
    @Entrypoint
    @staticmethod
    def parse_with_format(date_str: str, format: str) -> float:
        year, month, day, hour, minute, second = -1, -1, -1, 0, 0, 0
        tz_str = ""

        format_str_len = len(format)
        date_str = date_str.lower()
        ampm = ""
        date_str_cursor = format_cursor = 0

        while format_cursor < format_str_len:
            if format[format_cursor] == "%" and format_cursor + 1 < format_str_len:
                directive = format[format_cursor + 1]
                format_cursor += 1

                if directive == "Y":
                    if is4d(date_str[date_str_cursor:date_str_cursor + 4]):
                        year = int(date_str[date_str_cursor:date_str_cursor + 4])
                        date_str_cursor += 4
                    else:
                        raise ValueError("Bad value for %Y:", date_str)
                elif directive == "m":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        month = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %m:", date_str)
                elif directive == "d":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        day = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %d:", date_str)
                elif directive == "H":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        hour = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %H:", date_str)
                elif directive == "I":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        hour = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %I:", date_str)
                elif directive == "M":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        minute = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %M:", date_str)
                elif directive == "S":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        second = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %SL", date_str)
                elif directive == "b":
                    month = date_str[date_str_cursor:date_str_cursor + 3]
                    if month in MONTH_TO_INT:
                        month = MONTH_TO_INT[month]
                        date_str_cursor += 3
                    else:
                        raise ValueError("Bad value for %b:", date_str)
                elif directive == "B":
                    # september
                    if date_str[date_str_cursor:date_str_cursor + 9] in MONTH_TO_INT:
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 9]
                        ]
                        date_str_cursor += 9
                    # february, november, december
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 8] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 8]
                        ]
                        date_str_cursor += 8
                    # january,october
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 7] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 7]
                        ]
                        date_str_cursor += 7
                    # august
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 6] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 6]
                        ]
                        date_str_cursor += 6
                    # march,april
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 5] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 5]
                        ]
                        date_str_cursor += 5
                    # june, july
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 4] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 4]
                        ]
                        date_str_cursor += 4
                    # may
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 3] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 3]
                        ]
                        date_str_cursor += 3
                    else:
                        raise ValueError("Bad value for %B:", date_str)
                elif directive == "Z":
                    # 5 character tz abbreviations (future proofing since we don't currently support any)
                    if TimeZoneChecker.isValidTimezone(
                        date_str[date_str_cursor:date_str_cursor + 4]
                    ):
                        tz_str = date_str[date_str_cursor:date_str_cursor + 4]
                        date_str_cursor += 4
                    # 4 character tz abbreviations (future proofing since we don't currenlty support any)
                    elif TimeZoneChecker.isValidTimezone(
                        date_str[date_str_cursor:date_str_cursor + 4]
                    ):
                        tz_str = date_str[date_str_cursor:date_str_cursor + 4]
                        date_str_cursor += 4
                    # e.g. EST, EDT, PST
                    elif TimeZoneChecker.isValidTimezone(
                        date_str[date_str_cursor:date_str_cursor + 3]
                    ):
                        tz_str = date_str[date_str_cursor:date_str_cursor + 3]
                        date_str_cursor += 3
                    # e.g. PT, ET, CT
                    elif TimeZoneChecker.isValidTimezone(
                        date_str[date_str_cursor:date_str_cursor + 2]
                    ):
                        tz_str = date_str[date_str_cursor:date_str_cursor + 2]
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %Z:", date_str)
                elif directive == "z":
                    # [+|-]DDDD or [+|-]DD:DD, e.g. +0000, +1200
                    if TimeZoneChecker.isValidTimezone(
                        date_str[date_str_cursor:date_str_cursor + 5]
                    ):
                        tz_str = date_str[date_str_cursor:date_str_cursor + 5]
                        date_str_cursor += 5
                    # [+|-]DD or [+|-]DD
                    elif TimeZoneChecker.isValidTimezone(
                        date_str[date_str_cursor:date_str_cursor + 3]
                    ):
                        tz_str = date_str[date_str_cursor:date_str_cursor + 3]
                        date_str_cursor += 3
                    else:
                        raise ValueError("Bad value for %z:", date_str)
                elif directive == "p":
                    tok = date_str[date_str_cursor:date_str_cursor + 2]
                    if tok != AM and tok != PM:
                        raise ValueError("Bad value for %p:", date_str)
                    ampm = tok
                    date_str_cursor += 2
                else:
                    raise ValueError("Unsupported directive:", directive)

                format_cursor += 1
            else:
                format_cursor += 1
                date_str_cursor += 1

        if ampm != "":
            if hour > 12 or hour < 1:
                raise ValueError("AM/PM specified. hour must be between 1 and 12")
            if ampm == AM and hour == 12:
                hour = 0
            elif ampm == PM:
                hour = hour + 12

        if not Chrono.is_valid_date(year, month, day):
            raise ValueError("Invalid date:", date_str)
        if not Chrono.is_valid_time(hour, minute, second):
            raise ValueError("Invalid time:", date_str)

        if month == -1:
            month = 1
        if day == -1:
            day = 1

        timezone = DateParser.get_timezone_from_string(tz_str)

        datetime = DateTime(
            date=Date(year=year, month=month, day=day),
            timeOfDay=TimeOfDay(hour=hour, minute=minute, second=second),
        )
        return timezone.timestamp(datetime)

    @Entrypoint
    @staticmethod
    def parse_with_timezone(date_str: str, timezone: TimeZone, format: str = "") -> float:
        if format != "":
            return DateParser.parse_with_format_and_timezone(date_str, format, timezone)
        try:
            return DateParser.parse_iso_str_and_timezone(date_str, timezone)
        except ValueError:
            return DateParser.parse_non_iso(date_str, timezone)

    @Entrypoint
    @staticmethod
    def parse(date_str: str, format: str = "") -> float:
        """
        Parse a date string and return a unix timestamp
        Parameters:
            date_str (str): A string representing a date time. examples: 2022-01-03T02:45 or January 2, 1997 2:00pm
            format (str): An optional format string. E.g. '%Y-%m-%d'. If no format string is provided, the parser will
                        correctly parse ISO 8601 formatted strings and a number of non ISO 8601 formats
        Returns:
            (float) A unix timestamp
        """
        if format != "":
            return DateParser.parse_with_format(date_str, format)

        try:
            return DateParser.parse_iso_str(date_str)
        except ValueError:
            return DateParser.parse_non_iso(date_str)

    @Entrypoint
    @staticmethod
    def parse_with_format_and_timezone(date_str: str, format: str, timezone: TimeZone) -> float:
        """
        Parse a date string in the specified format and return a unix timestamp
        Parameters:
            date_str (str): A date string
            format (str): A string containing format directives. E.g. '%Y-%m-%d'
                Supported directives are:
                   %Y (zero padded 4 digit year)
                   %m (zero padded 2 digit month, 01-12)
                   %d (zero padded 2 digit day, 01-31)
                   %H (zero padded 2 digit hour in 24 hour format, 00-24)
                   %I (zero padded 2 digit hour in 12 hour format, 00-12)
                   %M (zero padded 2 digit minute, 00-59)
                   %S (zero padded 2 digit second, 00-59)
                   %b (3 character month abbreviation, jan-dec)
                   %B (month, january-december)
                   %Z (timezone abbreviation, e.g. EST, UTC, NYC)
                   %z (timezone offset, e.g. +0000, +00:00)
        Returns:
            (float) A unix timestamp
        """
        year, month, day, hour, minute, second = -1, -1, -1, 0, 0, 0

        format_str_len = len(format)
        date_str = date_str.lower()
        ampm = ""
        date_str_cursor = format_cursor = 0

        while format_cursor < format_str_len:
            if format[format_cursor] == "%" and format_cursor + 1 < format_str_len:
                directive = format[format_cursor + 1]
                format_cursor += 1

                if directive == "Y":
                    if is4d(date_str[date_str_cursor:date_str_cursor + 4]):
                        year = int(date_str[date_str_cursor:date_str_cursor + 4])
                        date_str_cursor += 4
                    else:
                        raise ValueError("Bad value for %Y:", date_str)
                elif directive == "m":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        month = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %m:", date_str)
                elif directive == "d":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        day = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %d:", date_str)
                elif directive == "H":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        hour = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %H:", date_str)
                elif directive == "I":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        hour = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %I:", date_str)
                elif directive == "M":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        minute = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %M:", date_str)
                elif directive == "S":
                    if is2d(date_str[date_str_cursor:date_str_cursor + 2]):
                        second = int(date_str[date_str_cursor:date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError("Bad value for %SL", date_str)
                elif directive == "b":
                    month = date_str[date_str_cursor:date_str_cursor + 3]
                    if month in MONTH_TO_INT:
                        month = MONTH_TO_INT[month]
                        date_str_cursor += 3
                    else:
                        raise ValueError("Bad value for %b:", date_str)
                elif directive == "B":
                    # september
                    if date_str[date_str_cursor:date_str_cursor + 9] in MONTH_TO_INT:
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 9]
                        ]
                        date_str_cursor += 9
                    # february, november, december
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 8] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 8]
                        ]
                        date_str_cursor += 8
                    # january,october
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 7] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 7]
                        ]
                        date_str_cursor += 7
                    # august
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 6] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 6]
                        ]
                        date_str_cursor += 6
                    # march,april
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 5] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 5]
                        ]
                        date_str_cursor += 5
                    # june, july
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 4] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 4]
                        ]
                        date_str_cursor += 4
                    # may
                    elif (
                        date_str[date_str_cursor:date_str_cursor + 3] in MONTH_TO_INT
                    ):
                        month = MONTH_TO_INT[
                            date_str[date_str_cursor:date_str_cursor + 3]
                        ]
                        date_str_cursor += 3
                    else:
                        raise ValueError("Bad value for %B:", date_str)

                elif directive.lower() == "z":
                    raise Exception(
                        "You cannot pass a timezone via string "
                        "if you are already passing it directly"
                    )

                elif directive == "p":
                    tok = date_str[date_str_cursor:date_str_cursor + 2]
                    if tok != AM and tok != PM:
                        raise ValueError("Bad value for %p:", date_str)
                    ampm = tok
                    date_str_cursor += 2
                else:
                    raise ValueError("Unsupported directive:", directive)

                format_cursor += 1
            else:
                format_cursor += 1
                date_str_cursor += 1

        if ampm != "":
            if hour > 12 or hour < 1:
                raise ValueError("AM/PM specified. hour must be between 1 and 12")
            if ampm == AM and hour == 12:
                hour = 0
            elif ampm == PM:
                hour = hour + 12

        if not Chrono.is_valid_date(year, month, day):
            raise ValueError("Invalid date:", date_str)
        if not Chrono.is_valid_time(hour, minute, second):
            raise ValueError("Invalid time:", date_str)

        if month == -1:
            month = 1
        if day == -1:
            day = 1

        datetime = DateTime(year, month, day, hour, minute, second)
        return timezone.timestamp(datetime)

    @Entrypoint
    @staticmethod
    def parse_iso_str_and_timezone(date_str: str, timezone: TimeZone) -> float:
        tokens = DateParser._get_tokens(
            time_str=date_str.lower().replace(" ", T), skip_chars="/-:"
        )

        # Process date segment
        date_tokens = ListOf(str)()
        cursor = 0
        while cursor < len(tokens):
            if tokens[cursor] == T or tokens[cursor] == PLUS or tokens[cursor] == DASH:
                cursor += 1
                break
            else:
                date_tokens.append(tokens[cursor])
                cursor += 1

        year = month = day = -1

        if len(date_tokens) == 1:
            if len(date_tokens[0]) == 8:
                year, month, day = (
                    int(date_tokens[0][:4]),
                    int(date_tokens[0][4:6]),
                    int(date_tokens[0][6:8]),
                )
            elif len(date_tokens[0]) == 6:
                year, month, day = int(date_tokens[0][:4]), int(date_tokens[0][4:6]), 1
            elif len(date_tokens[0]) == 4:
                year, month, day = int(date_tokens[0][:4]), 1, 1
        elif len(date_tokens) == 2 and is4d(date_tokens[0]):
            year, month, day = int(date_tokens[0]), int(date_tokens[1]), 1
        elif len(date_tokens) == 3 and is4d(date_tokens[0]):
            year, month, day = (
                int(date_tokens[0]),
                int(date_tokens[1]),
                int(date_tokens[2]),
            )
        else:
            raise ValueError("Invalid format: ", date_tokens)

        if not Chrono.is_valid_date(year, month, day):
            raise ValueError("Invalid date_tokens: ", date_tokens)

        midnight = timezone.timestamp(DateTime(year, month, day, 0, 0, 0.0))

        if cursor >= len(tokens):
            return midnight

        # Process time segement
        time_tokens = ListOf(str)()
        while cursor < len(tokens):
            if tokens[cursor] == T:
                cursor += 1
                break
            elif (
                tokens[cursor] == PLUS
                or tokens[cursor] == DASH
            ):
                break
            elif tokens[cursor].isalpha():
                raise ValueError("Invalid date_tokens: ", date_tokens)
            else:
                time_tokens.append(tokens[cursor])
                cursor += 1

        timeOfDay = DateParser._parse_iso_time_tokens(time_tokens)
        datetime = DateTime(year, month, day, timeOfDay.hour, timeOfDay.minute, timeOfDay.second)

        return timezone.timestamp(datetime)

    @Entrypoint
    @staticmethod
    def parse_iso_str(date_str: str) -> float:
        """
        Converts an ISO 8601 formated date string to a unix timestamp
        (this means year - month - day)
        Parameters:
            date_str (str): An ISO 8601 formatted string
        Returns:
            unixtime(float): A unix timestamp
        """
        tokens = DateParser._get_tokens(
            time_str=date_str.lower().replace(" ", T), skip_chars="/-:"
        )

        # Process date segment
        date_tokens = ListOf(str)()
        cursor = 0
        while cursor < len(tokens):
            if tokens[cursor] == T or tokens[cursor] == PLUS or tokens[cursor] == DASH:
                cursor += 1
                break
            else:
                date_tokens.append(tokens[cursor])
                cursor += 1

        year = month = day = -1

        if len(date_tokens) == 1:
            if len(date_tokens[0]) == 8:
                year, month, day = (
                    int(date_tokens[0][:4]),
                    int(date_tokens[0][4:6]),
                    int(date_tokens[0][6:8]),
                )
            elif len(date_tokens[0]) == 6:
                year, month, day = int(date_tokens[0][:4]), int(date_tokens[0][4:6]), 1
            elif len(date_tokens[0]) == 4:
                year, month, day = int(date_tokens[0][:4]), 1, 1
        elif len(date_tokens) == 2 and is4d(date_tokens[0]):
            year, month, day = int(date_tokens[0]), int(date_tokens[1]), 1
        elif len(date_tokens) == 3 and is4d(date_tokens[0]):
            year, month, day = (
                int(date_tokens[0]),
                int(date_tokens[1]),
                int(date_tokens[2]),
            )
        else:
            raise ValueError("Invalid format: ", date_tokens)

        if not Chrono.is_valid_date(year, month, day):
            raise ValueError("Invalid date_tokens: ", date_tokens)

        midnight = UTC.timestamp(DateTime(year, month, day, 0, 0, 0.0))

        if cursor >= len(tokens):
            return midnight

        # Process time segement
        time_tokens = ListOf(str)()
        while cursor < len(tokens):
            if tokens[cursor] == T:
                cursor += 1
                break
            elif (
                tokens[cursor] == PLUS
                or tokens[cursor] == DASH
                or tokens[cursor].isalpha()
            ):
                break
            else:
                time_tokens.append(tokens[cursor])
                cursor += 1

        timeOfDay = DateParser._parse_iso_time_tokens(time_tokens)
        datetime = DateTime(year, month, day, timeOfDay.hour, timeOfDay.minute, timeOfDay.second)

        if cursor >= len(tokens):
            return UTC.timestamp(datetime)

        timezone = DateParser.get_timezone_from_string("".join(tokens[cursor:]))

        return timezone.timestamp(datetime)

    @Entrypoint
    @staticmethod
    def _parse_iso_time_tokens(time_tokens: ListOf(str)) -> TimeOfDay:
        hour, minute, second = 0, 0, 0.0

        if len(time_tokens) == 1:
            if len(time_tokens[0]) == 6:
                hour, minute, second = (
                    int(time_tokens[0][:2]),
                    int(time_tokens[0][2:4]),
                    float(time_tokens[0][4:6]),
                )
            elif len(time_tokens[0]) == 4:
                hour, minute, second = (
                    int(time_tokens[0][:2]),
                    int(time_tokens[0][2:4]),
                    0.0,
                )
            elif len(time_tokens[0]) == 2:
                hour, minute, second = int(time_tokens[0][:2]), 0, 0.0
        elif len(time_tokens) == 2:
            hour, minute, second = int(time_tokens[0]), int(time_tokens[1]), 0.0
        elif len(time_tokens) == 3:
            hour, minute, second = (
                int(time_tokens[0]),
                int(time_tokens[1]),
                float(time_tokens[2]),
            )

        if not Chrono.is_valid_time(hour, minute, second):
            raise ValueError("Invalid time: ", time_tokens)

        return TimeOfDay(hour=hour, minute=minute, second=second)

    @Entrypoint
    @staticmethod
    def _get_tokens(time_str: str, skip_chars: str = "") -> ListOf(str):
        """
        Tokenises a string into components suitable for datetime processing
        Parameters:
            time_str (str): A string representing a date time. examples: 2022-01-03T02:45 or January 2, 1997 2:00pm
        Returns:
            (ListOf(str) A list of tokens. E.g. ["1997", "/", "01", "/", "02"]
        """

        tokens = ListOf(str)()
        cursor = 0

        strlen = len(time_str)
        while cursor < strlen:
            # greedily grab all alpha characters to make a string token
            start = stop = cursor

            while cursor < strlen and time_str[cursor].isalpha():
                cursor += 1
                stop += 1

            if time_str[start:stop]:
                tokens.append(time_str[start:stop])

            # greedily grab digits or . to make a numeric token
            start = stop = cursor
            while cursor < strlen and (
                time_str[cursor].isdigit() or time_str[cursor] == DOT
            ):
                cursor += 1
                stop += 1
                continue

            if time_str[start:stop]:
                tokens.append(time_str[start:stop])

            # whitespace
            start = stop = cursor
            while cursor < strlen and time_str[cursor].isspace():
                cursor += 1
                stop += 1

            if not skip_chars.find(SPACE) > -1 and time_str[start:stop]:
                tokens.append(SPACE)

            # everything else is a separator (e.g. / or :)
            start = stop = cursor
            while (
                cursor < strlen
                and not time_str[cursor].isspace()
                and not time_str[cursor].isdigit()
                and not time_str[cursor].isalpha()
            ):
                cursor += 1
                stop += 1

            if not skip_chars.find(time_str[start:stop]) > -1 and time_str[start:stop]:
                tokens.append(time_str[start:stop])
        return tokens

    @Entrypoint
    @staticmethod
    def parse_non_iso_time(tokens) -> TimeOfDay:
        """
        Converts a set of tokens representing a time seconds
        Parameters:
            tokens (str): The time tokens
        Returns:
            (float): The seconds
        """
        h = m = s = 0

        if len(tokens) == 0:
            return TimeOfDay(hour=0, minute=0, second=0.0)

        # break into time and ampm parts
        sep_idx = -1
        for idx, token in enumerate(tokens):
            if token == AM or token == PM:
                sep_idx = idx
                break

        if sep_idx > -1:
            time_part, ampm_part = tokens[:sep_idx], tokens[sep_idx:]
        else:
            time_part = tokens
            ampm_part = ListOf(str)()

        if (
            len(time_part) == 5
            and time_part[1] == COLON
            and time_part[2].isdigit()
            and time_part[3] == COLON
        ):
            # HH:MM:SS
            if time_part[0].isdigit() and time_part[4].isdigit():
                h, m, s = int(time_part[0]), int(time_part[2]), int(time_part[4])
            else:
                raise ValueError()
        elif len(time_part) == 3:
            # HH:MM
            if (
                time_part[0].isdigit()
                and time_part[1] == COLON
                and time_part[2].isdigit()
            ):
                h, m, s = int(time_part[0]), int(time_part[2]), 0
            else:
                raise ValueError()
        else:
            raise ValueError("Unsupported time format", tokens)

        if len(ampm_part) > 0:
            if h > 12 or h < 1:
                raise ValueError("AM/PM specified. hour must be between 1 and 12")
            if ampm_part[0] == AM and h == 12:
                h = 0
            elif ampm_part[0] == PM:
                h = h + 12

        if not Chrono.is_valid_time(h, m, s):
            raise ValueError("Invalid time: ", h, m, s)
        return TimeOfDay(hour=h, minute=m, second=s)

    @Entrypoint
    @staticmethod
    def parse_non_iso(date_str: str, timezone: TimeZone = UTC) -> float:
        """
        Parse a date string and return a unix timestamp
        Parameters:
            date_str (str): A date string
        Returns:
            (float) A unix timestamp
        """
        date_str = date_str.lower().replace("a.m.", AM).replace("p.m.", PM)
        tokens = DateParser._get_tokens(date_str, skip_chars=" ,")

        # if/elsif block is long but it's simple and allows us to clearly define the formats we support
        # and add new formats as needed

        y = m = d = 0
        time_tokens = ListOf(str)()

        # 5+ tokens with 4 digit year as 5th token
        if len(tokens) >= 5 and is4d(tokens[4]):
            # DD/Month/YYYY or DD-Month-YYYY
            if is_month(tokens[2]) and (
                (tokens[1] == BACKSLASH and tokens[3] == BACKSLASH)
                or (tokens[1] == DASH and tokens[3] == DASH)
            ):
                y, m, d, time_tokens = (
                    int(tokens[4]),
                    MONTH_TO_INT[tokens[2]],
                    int(tokens[0]),
                    tokens[5:],
                )

            # Month-DD-YYYY
            elif (
                is_month(tokens[0])
                and tokens[1] == DASH
                and tokens[2].isdigit()
                and tokens[3] == DASH
            ):
                y, m, d, time_tokens = (
                    int(tokens[4]),
                    MONTH_TO_INT[tokens[0]],
                    int(tokens[2]),
                    tokens[5:],
                )

            # Month-DD-YYYY or Month/DD/YYYY
            elif is_month(tokens[0]) and (
                (tokens[1] == BACKSLASH and tokens[3] == BACKSLASH)
                or (tokens[1] == DASH and tokens[3] == DASH)
            ):
                y, m, d, time_tokens = (
                    int(tokens[4]),
                    MONTH_TO_INT[tokens[0]],
                    int(tokens[2]),
                    tokens[5:],
                )

            # 5+ DD-MM-YYYY or DD/MM/YYYY
            elif (
                len(tokens) >= 5
                and is4d(tokens[4])
                and (tokens[1] == BACKSLASH and tokens[3] == BACKSLASH)
                or (tokens[1] == DASH and tokens[3] == DASH)
                and tokens[0].isdigit()
                and tokens[2].isdigit()
            ):
                if int(tokens[0]) > 12:
                    y, m, d, time_tokens = (
                        int(tokens[4]),
                        int(tokens[2]),
                        int(tokens[0]),
                        tokens[5:]
                    )
                else:
                    y, m, d, time_tokens = (
                        int(tokens[4]),
                        int(tokens[0]),
                        int(tokens[2]),
                        tokens[5:]
                    )
            else:
                raise ValueError("Unsupported date format: " + date_str)

        # YYYY/Month/DD or YYYY-Month-DD
        elif (
            len(tokens) >= 5
            and is4d(tokens[0])
            and is_month(tokens[2])
            and tokens[4].isdigit()
            and (
                (tokens[1] == BACKSLASH and tokens[3] == BACKSLASH)
                or ( tokens[1] == DASH and tokens[3] == DASH)
            )
        ):
            y, m, d, time_tokens = (
                int(tokens[0]),
                MONTH_TO_INT[tokens[2]],
                int(tokens[4]),
                tokens[5:],
            )

        # Month D YYYY
        elif (
            len(tokens) >= 3
            and is_month(tokens[0])
            and tokens[1].isdigit()
            and is4d(tokens[2])
        ):
            y, m, d, time_tokens = (
                int(tokens[2]),
                MONTH_TO_INT[tokens[0]],
                int(tokens[1]),
                tokens[3:],
            )

        # D Month YYYY
        elif (
            len(tokens) >= 3
            and is_month(tokens[1])
            and tokens[0].isdigit()
            and is4d(tokens[2])
        ):
            y, m, d, time_tokens = (
                int(tokens[2]),
                MONTH_TO_INT[tokens[1]],
                int(tokens[0]),
                tokens[3:],
            )

        # YYYY Month DD
        elif (
            len(tokens) >= 3
            and is4d(tokens[0])
            and is_month(tokens[1])
            and tokens[2].isdigit()
        ):
            y, m, d, time_tokens = (
                int(tokens[0]),
                MONTH_TO_INT[tokens[1]],
                int(tokens[2]),
                tokens[3:],
            )

        # YYYY-MM-DD or YYYY/MM/DD
        elif (
            len(tokens) >= 5
            and is4d(tokens[0])
            and tokens[2].isdigit()
            and tokens[4].isdigit()
            and (
                (tokens[1] == BACKSLASH and tokens[3] == BACKSLASH)
                or (tokens[1] == DASH and tokens[3] == DASH)
            )
        ):
            y, m, d, time_tokens = (
                int(tokens[0]),
                int(tokens[2]),
                int(tokens[4]),
                tokens[5:],
            )

        # YYYYMMDD-time
        elif (
            len(tokens) >= 1
            and len(tokens[0]) == 8
            and tokens[0].isdigit()
            and tokens[1] == DASH
        ):
            y, m, d, time_tokens = (
                int(''.join(tokens[0][:4])),
                int(''.join(tokens[0][4:6])),
                int(''.join(tokens[0][6:8])),
                tokens[2:],
            )

        # DD-Month-YY or DD/Month/YY
        elif (
            len(tokens) >= 5
            and is2d(tokens[0])
            and is2d(tokens[4])
            and is_month(tokens[2])
            and (
                (tokens[1] == BACKSLASH and tokens[3] == BACKSLASH)
                or (tokens[1] == DASH and tokens[3] == DASH)
            )
        ):
            y, m, d, time_tokens = (
                int(tokens[4]),
                MONTH_TO_INT[tokens[2]],
                int(tokens[0]),
                tokens[5:],
            )

            y = y + (1900 if y >= 1900 else 2000)

        else:
            raise ValueError("Unsupported date format: " + date_str)

        if not Chrono.is_valid_date(y, m, d):
            raise ValueError("Invalid date: " + date_str)

        timeOfDay = DateParser.parse_non_iso_time(time_tokens)
        datetime = DateTime(y, m, d, timeOfDay.hour, timeOfDay.minute, timeOfDay.second)
        return timezone.timestamp(datetime)

    @Entrypoint
    @staticmethod
    def get_timezone_from_string(tz_str: str) -> TimeZone:
        if TimeZoneChecker.isValidTimezone(tz_str):
            return TimeZoneChecker.TIMEZONES[tz_str]

        elif tz_str[0] == "+" or tz_str[0] == "-":
            tz_components = tz_str[1:]
            if len(tz_components) == 2:
                offset_hours = float(tz_components)
            elif len(tz_components) == 4:
                offset_hours = float(tz_components[:2]) + float(tz_components[2:]) / 60
            elif len(tz_components) == 6:
                offset_hours = (
                    float(tz_components[:2])
                    + float(tz_components[2:4]) / 60
                    + float(tz_components[4:]) / 3600
                )
            else:
                raise ValueError("Invalid timezone offset: ", tz_str)

            if tz_str[0] == "-":
                offset_hours = -offset_hours

            return FixedOffsetTimezone(offset_hours=offset_hours)

        raise ValueError("Invalid timezone offset: ", tz_str)
