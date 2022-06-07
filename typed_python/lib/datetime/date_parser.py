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

from typed_python import Class, Final
from typed_python import Entrypoint, ListOf, Dict
from typed_python.lib.datetime.timezone import tz_abbr_to_utc_offset, tz_string_to_utc_offset, is_supported_tz_abbr
from typed_python.lib.datetime.chrono import time_to_seconds, date_to_seconds, is_date, is_time

JAN = 'jan'
FEB = 'feb'
MAR = 'mar'
APR = 'apr'
MAY = 'may'
JUN = 'jun'
JUL = 'jul'
AUG = 'aug'
SEP = 'sep'
OCT = 'oct'
NOV = 'nov'
DEC = 'dec'

JANUARY = 'january'
FEBRUARY = 'february'
MARCH = 'march'
APRIL = 'april'
# MAY = 'may'
JUNE = 'june'
JULY = 'july'
AUGUST = 'august'
SEPT = 'sept'
SEPTEMBER = 'september'
OCTOBER = 'october'
NOVEMBER = 'november'
DECEMBER = 'december'

MONTHS = Dict(str, int)({
    JAN: 1, JANUARY: 1,
    FEB: 2, FEBRUARY: 2,
    MAR: 3, MARCH: 3,
    APR: 4, APRIL: 4,
    MAY: 5,
    JUN: 6, JUNE: 6,
    JUL: 7, JULY: 7,
    AUG: 8, AUGUST: 8,
    SEP: 9, SEPT: 9, SEPTEMBER: 9,
    OCT: 10, OCTOBER: 10,
    NOV: 11, NOVEMBER: 11,
    DEC: 12, DECEMBER: 12
})

T = 'T'
PLUS = '+'
COLON = ':'
BACKSLASH = '/'
DASH = '-'
COMMA = ','
SPACE = ' '
DOT = '.'
AM = 'am'
PM = 'pm'


class DateParser(Class, Final):

    @Entrypoint
    def is_tz_offset(hour: int, min: int, second: float = 0.0) -> bool:
        '''
        Tests if an hour,min combination is a valid offset from UTC
        Parameters:
            hour(int): The hour
            min(int): The minute
        Returns:
            True if the inputs are in the range UTC-12:00 to UTC+14
        '''
        if hour > 14 or hour < -12:
            return False

        if (hour == 14 or hour == -12) and min > 0:
            return False

        if min < 0 or min + second >= 60:
            return False

        return True

    @Entrypoint
    @staticmethod
    def is_month(month_str: str) -> bool:
        '''
        Tests if string represents a valid month
        Parameters:
            month_str: The month string (case insenstive). Examples: 'Jan', 'January'
        Returns:
            True if the month string is valid, False otherwise
        '''
        return month_str.strip().lower() in MONTHS

    @Entrypoint
    @staticmethod
    def is4d(str: str) -> bool:
        '''
        Tests if a string is 4 digits.
        Parameters:
            str(string):
        Returns:
            True if the input is a 4 digit string
        '''
        return len(str) == 4 and str.isdigit()

    @Entrypoint
    @staticmethod
    def is2d(str: str) -> bool:
        '''
        Tests if a string is 2 digits.
        Parameters:
            str(string):
        Returns:
            True if the input is a 2 digit string
        '''
        return len(str) == 2 and str.isdigit()

    @Entrypoint
    @staticmethod
    def is_year(year_str: str) -> bool:
        '''
        Tests if a string represents a valid 4 digit year.
        Parameters:
            year(string): The year
        Returns:
            True if the input is a 4 digit string
        '''
        return len(year_str) == 4 and year_str.isdigit()

    @Entrypoint
    @staticmethod
    def parse(date_str: str) -> float:
        '''
        Parse a date string and return a unix timestamp
        Parameters:
            date_str (str): A string representing a date time. examples: 2022-01-03T02:45 or January 2, 1997 2:00pm
        Returns:
            (float) A unix timestamp
        '''
        try:
            return DateParser.parse_iso(date_str)
        except ValueError:
            return DateParser.parse_non_iso(date_str)

    @Entrypoint
    @staticmethod
    def get_tokens(time_str: str, skip_chars: str = '') -> ListOf(str):
        '''
        Tokenises a string into components suitable for datetime processing
        Parameters:
            time_str (str): A string representing a date time. examples: 2022-01-03T02:45 or January 2, 1997 2:00pm
        Returns:
            (ListOf(str) A list of tokens. E.g. ["1997", "/", "01", "/", "02"]
        '''

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
            while cursor < strlen and (time_str[cursor].isdigit() or time_str[cursor] == DOT):
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
            while (cursor < strlen and not time_str[cursor].isspace()
                   and not time_str[cursor].isdigit() and not time_str[cursor].isalpha()):
                cursor += 1
                stop += 1

            if not skip_chars.find(time_str[start:stop]) > -1 and time_str[start:stop]:
                tokens.append(time_str[start:stop])
        return tokens

    @Entrypoint
    @staticmethod
    def parse_tz_offset(tokens: ListOf(str)) -> int:
        '''
        Converts a set of tokens representing a timezone offset to seconds.
        Parameters:
            tokens (ListOf(str)): A set of string tokens representing a timezone. E.g. ['Z'] or ['+', '02', COLON, '23']
        Returns:
            (int): The offset in seconds
        '''
        if tokens[0] != PLUS and tokens[0] != DASH:
            raise ValueError("tz offset must begin with '+' or DASH")

        # Note
        # You'll see this pattern
        #    x, y, z = a, b, c
        # instead of the possibly more aesthetic
        #    x = a
        #    y = b
        #    z = c
        # in a lot of places in this file. It's only like this because, for reasons unclear, the former
        # appears to have a slight, but noticable performance advantage over the latter.
        hour, min, second = 0, 0, 0.0

        if len(tokens) == 2:
            # [+|-]HH or [+|-]HHMM
            if len(tokens[1]) == 2:
                hour, min, second = int(tokens[1]), 0, 0.0
            elif len(tokens[1]) == 4:
                hour, min, second = int(tokens[1][:2]), int(tokens[1][2:4]), 0.0
            elif len(tokens[1]) >= 6:
                hour, min, second = int(tokens[1][:2]), int(tokens[1][2:4]), float(tokens[1][6:])
        elif len(tokens) == 3:
            hour, min, second = int(tokens[1]), int(tokens[2]), 0.0
        elif len(tokens) == 4:
            hour, min, second = int(tokens[1]), int(tokens[2]), float(tokens[2])
        else:
            raise ValueError('Invalid tz offset')

        hour = hour * -1 if tokens[0] == DASH else hour

        if DateParser.is_tz_offset(hour, min, second):
            return hour * 3600 + (min * 60 if hour > 0 else min * -60) + (second if hour > 0 else second * -1)
        else:
            raise ValueError('Invalid tz offset: ')

    @Entrypoint
    @staticmethod
    def parse_iso_time(time_tokens: ListOf(str)):
        hour, minute, second = 0, 0, 0.0

        if len(time_tokens) == 1:
            if len(time_tokens[0]) == 6:
                hour, minute, second = int(time_tokens[0][:2]), int(time_tokens[0][2:4]), float(time_tokens[0][4:6])
            elif len(time_tokens[0]) == 4:
                hour, minute, second = int(time_tokens[0][:2]), int(time_tokens[0][2:4]), 0.0
            elif len(time_tokens[0]) == 2:
                hour, minute, second = int(time_tokens[0][:2]), 0, 0.0
        elif len(time_tokens) == 2:
            hour, minute, second = int(time_tokens[0]), int(time_tokens[1]), 0.0
        elif len(time_tokens) == 3:
            hour, minute, second = int(time_tokens[0]), int(time_tokens[1]), float(time_tokens[2])

        if not is_time(hour, minute, second):
            raise ValueError('Invalid time: ', time_tokens)

        return time_to_seconds(hour, minute, second)

    @Entrypoint
    @staticmethod
    def parse_iso(date_str: str) -> float:
        '''
        Converts an ISO 8601 formated date string to a unix timestamp
        Parameters:
            date_str (str): An ISO 8601 formatted string
        Returns:
            unixtime(float): A unix timestamp
        '''
        tokens = DateParser.get_tokens(time_str=date_str.upper().replace(' ', T), skip_chars='/-:')

        # Process date segment
        date_tokens = ListOf(str)()
        cursor = 0
        while cursor < len(tokens):
            if (tokens[cursor] == T or tokens[cursor] == PLUS or tokens[cursor] == DASH):
                cursor += 1
                break
            else:
                date_tokens.append(tokens[cursor])
                cursor += 1

        year = month = day = -1

        if len(date_tokens) == 1:
            if len(date_tokens[0]) == 8:
                year, month, day = int(date_tokens[0][:4]), int(date_tokens[0][4:6]), int(date_tokens[0][6:8])
            elif len(date_tokens[0]) == 6:
                year, month, day = int(date_tokens[0][:4]), int(date_tokens[0][4:6]), 1
            elif len(date_tokens[0]) == 4:
                year, month, day = int(date_tokens[0][:4]), 1, 1
        elif len(date_tokens) == 2 and DateParser.is4d(date_tokens[0]):
            year, month, day = int(date_tokens[0]), int(date_tokens[1]), 1
        elif len(date_tokens) == 3 and DateParser.is4d(date_tokens[0]):
            year, month, day = int(date_tokens[0]), int(date_tokens[1]), int(date_tokens[2])
        else:
            raise ValueError('Invalid format: ', date_tokens)

        if not is_date(year, month, day):
            raise ValueError('Invalid date_tokens: ', date_tokens)

        dt = date_to_seconds(year, month, day)

        if cursor >= len(tokens):
            return dt

        # Process time segement
        time_tokens = ListOf(str)()
        while cursor < len(tokens):
            if tokens[cursor] == T:
                cursor += 1
                break
            elif tokens[cursor] == PLUS or tokens[cursor] == DASH or tokens[cursor].isalpha():
                break
            else:
                time_tokens.append(tokens[cursor])
                cursor += 1

        dt += DateParser.parse_iso_time(time_tokens)
        if cursor >= len(tokens):
            return dt

        tz_tokens = ListOf(str)()
        while cursor < len(tokens):
            tz_tokens.append(tokens[cursor])
            cursor += 1

        if len(tz_tokens) == 1:
            tz_offset = tz_string_to_utc_offset(tz_tokens[0], dt)
        elif tz_tokens[0] == PLUS or tz_tokens[0] == DASH:
            tz_offset = DateParser.parse_tz_offset(tz_tokens)
        else:
            raise ValueError('Unsupported tz format', tz_tokens)

        return dt + tz_offset

    @ Entrypoint
    @ staticmethod
    def parse_non_iso_time(tokens) -> float:
        '''
        Converts a set of tokens representing a time seconds
        Parameters:
            tokens (str): The time tokens
        Returns:
            (float): The seconds
        '''
        h = m = s = 0

        if len(tokens) == 0:
            return 0

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

        if len(time_part) == 5 and time_part[1] == COLON and time_part[2].isdigit() and time_part[3] == COLON:
            # HH:MM:SS
            if time_part[0].isdigit() and time_part[4].isdigit():
                h, m, s = int(time_part[0]), int(time_part[2]), int(time_part[4])
            else:
                raise ValueError()
        elif len(time_part) == 3:
            # HH:MM
            if time_part[0].isdigit() and time_part[1] == COLON and time_part[2].isdigit():
                h, m, s = int(time_part[0]), int(time_part[2]), 0
            else:
                raise ValueError()
        else:
            raise ValueError('Unsupported time format', tokens)

        if len(ampm_part) > 0:
            if h > 12 or h < 1:
                raise ValueError('AM/PM specified. hour must be between 1 and 12')
            if ampm_part[0].lower() == AM and h == 12:
                h = 0
            elif ampm_part[0].lower() == PM:
                h = h + 12

        if not is_time(h, m, s):
            raise ValueError('Invalid time: ', h, m, s)
        return time_to_seconds(h, m, s)

    @Entrypoint
    @staticmethod
    def parse_non_iso(date_str: str) -> float:
        '''
        Parse a date string and return a unix timestamp
        Parameters:
            date_str (str): A date string
        Returns:
            (float) A unix timestamp
        '''
        date_str = date_str.lower().replace('a.m.', AM).replace('p.m.', PM).strip()
        tokens = DateParser.get_tokens(date_str, skip_chars=' ,')

        # if/elsif block is long but it's simple and allows us to clearly define the formats we support
        # and add new formats as needed

        y = m = d = 0
        time_tokens = ListOf(str)()

        # 5+ tokens with 4 digit year as 5th token
        if len(tokens) >= 5 and DateParser.is4d(tokens[4]):
            # DD/Month/YYYY or DD-Month-YYYY
            if (DateParser.is_month(tokens[2]) and ((tokens[1] == BACKSLASH and tokens[3] == BACKSLASH) or
               (tokens[1] == DASH and tokens[3] == DASH))):
                y, m, d, time_tokens = int(tokens[4]), MONTHS[tokens[2].lower()], int(tokens[0]), tokens[5:]

            # Month-DD-YYYY
            elif DateParser.is_month(tokens[0]) and tokens[1] == DASH and tokens[2].isdigit() and tokens[3] == DASH:
                y, m, d, time_tokens = int(tokens[4]), MONTHS[tokens[0].lower()], int(tokens[2]), tokens[5:]

            # Month-DD-YYYY or Month/DD/YYYY
            elif (DateParser.is_month(tokens[0]) and ((tokens[1] == BACKSLASH and tokens[3] == BACKSLASH) or
                                                      (tokens[1] == DASH and tokens[3] == DASH))):
                y, m, d, time_tokens = int(tokens[4]), MONTHS[tokens[0].lower()], int(tokens[2]), tokens[5:]

            else:
                raise ValueError('Unsupported date format: ' + date_str)

        # 5+ tokens with 4 digit year as 1st token
        elif len(tokens) >= 5 and DateParser.is4d(tokens[0]) and DateParser.is_month(tokens[2]) and tokens[4].isdigit():
            # YYYY/Month/DD or YYYY-Month-DD
            if (tokens[1] == BACKSLASH and tokens[3] == BACKSLASH) or (tokens[1] == DASH and tokens[3] == DASH):
                y, m, d, time_tokens = int(tokens[0]), MONTHS[tokens[2].lower()], int(tokens[4]), tokens[5:]
            else:
                raise ValueError('Unsupported date format: ' + date_str)

        # Month D YYYY
        elif len(tokens) >= 3 and DateParser.is_month(tokens[0]) and tokens[1].isdigit() and DateParser.is4d(tokens[2]):
            y, m, d, time_tokens = int(tokens[2]), MONTHS[tokens[0].lower()], int(tokens[1]), tokens[3:]

        # D Month YYYY
        elif len(tokens) >= 3 and DateParser.is_month(tokens[1]) and tokens[0].isdigit() and DateParser.is4d(tokens[2]):
            y, m, d, time_tokens = int(tokens[2]), MONTHS[tokens[1].lower()], int(tokens[0]), tokens[3:]

        # YYYY Month DD
        elif len(tokens) >= 3 and DateParser.is4d(tokens[0]) and DateParser.is_month(tokens[1]) and tokens[2].isdigit():
            y, m, d, time_tokens = int(tokens[0]), MONTHS[tokens[1].lower()], int(tokens[2]), tokens[3:]

        else:
            raise ValueError('Unsupported date format: ' + date_str)

        if not is_date(y, m, d):
            raise ValueError('Invalid date: ' + date_str)

        return date_to_seconds(y, m, d) + DateParser.parse_non_iso_time(time_tokens)

    @Entrypoint
    @staticmethod
    def parse_format(date_str: str, format: str) -> float:
        '''
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
        '''
        year, month, day, hour, minute, second = -1, -1, -1, 0, 0, 0
        tz_str = ''

        format_str_len = len(format)
        date_str = date_str.lower()
        date_str_len = len(date_str)
        ampm = ''
        date_str_cursor = format_cursor = 0

        while format_cursor < format_str_len:
            if format[format_cursor] == '%' and format_cursor + 1 < format_str_len:
                directive = format[format_cursor + 1]
                format_cursor += 1

                if directive == 'Y':
                    if date_str_cursor + 4 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 4].isdigit():
                        year = int(date_str[date_str_cursor: date_str_cursor + 4])
                        date_str_cursor += 4
                    else:
                        raise ValueError('Bad value for %Y:', date_str)
                elif directive == 'm':
                    if date_str_cursor + 2 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 2].isdigit():
                        month = int(date_str[date_str_cursor: date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError('Bad value for %m:', date_str)
                elif directive == 'd':
                    if date_str_cursor + 2 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 2].isdigit():
                        day = int(date_str[date_str_cursor: date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError('Bad value for %d:', date_str)
                elif directive == 'H':
                    if date_str_cursor + 2 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 2].isdigit():
                        hour = int(date_str[date_str_cursor: date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError('Bad value for %H:', date_str)
                elif directive == 'I':
                    if date_str_cursor + 2 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 2].isdigit():
                        hour = int(date_str[date_str_cursor: date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError('Bad value for %I:', date_str)
                elif directive == 'M':
                    if date_str_cursor + 2 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 2].isdigit():
                        minute = int(date_str[date_str_cursor: date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError('Bad value for %M:', date_str)
                elif directive == 'S':
                    if date_str_cursor + 2 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 2].isdigit():
                        second = int(date_str[date_str_cursor: date_str_cursor + 2])
                        date_str_cursor += 2
                    else:
                        raise ValueError('Bad value for %SL', date_str)
                elif directive == 'b':
                    if date_str_cursor + 3 > date_str_len:
                        raise ValueError('Bad value for %b:', date_str)
                    month = date_str[date_str_cursor: date_str_cursor + 3]
                    if month in MONTHS:
                        month = MONTHS[month]
                        date_str_cursor += 3
                    else:
                        raise ValueError('Bad value for %b:', date_str)
                elif directive == 'B':
                    # september
                    if date_str_cursor + 9 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 9] in MONTHS:
                        month = MONTHS[date_str[date_str_cursor: date_str_cursor + 9]]
                        date_str_cursor += 9
                    # february, november, december
                    elif date_str_cursor + 8 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 8] in MONTHS:
                        month = MONTHS[date_str[date_str_cursor: date_str_cursor + 8]]
                        date_str_cursor += 8
                    # january,october
                    elif date_str_cursor + 7 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 7] in MONTHS:
                        month = MONTHS[date_str[date_str_cursor: date_str_cursor + 7]]
                        date_str_cursor += 7
                    # august
                    elif date_str_cursor + 6 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 6] in MONTHS:
                        month = MONTHS[date_str[date_str_cursor: date_str_cursor + 6]]
                        date_str_cursor += 6
                    # march,april
                    elif date_str_cursor + 5 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 5] in MONTHS:
                        month = MONTHS[date_str[date_str_cursor: date_str_cursor + 5]]
                        date_str_cursor += 5
                    # june, july
                    elif date_str_cursor + 4 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 4] in MONTHS:
                        month = MONTHS[date_str[date_str_cursor: date_str_cursor + 4]]
                        date_str_cursor += 4
                    # may
                    elif date_str_cursor + 3 <= date_str_len and date_str[date_str_cursor: date_str_cursor + 3] in MONTHS:
                        month = MONTHS[date_str[date_str_cursor: date_str_cursor + 3]]
                        date_str_cursor += 3
                    else:
                        raise ValueError('Bad value for %B:', date_str)
                elif directive == 'Z':
                    # 5 character tz abbreviations (future proofing since we don't currently support any)
                    if date_str_cursor + 4 <= date_str_len and is_supported_tz_abbr(date_str[date_str_cursor: date_str_cursor + 4]):
                        tz_str = date_str[date_str_cursor: date_str_cursor + 4]
                        date_str_cursor += 4
                    # 4 character tz abbreviations (future proofing since we don't currenlty support any)
                    elif date_str_cursor + 4 <= date_str_len and is_supported_tz_abbr(date_str[date_str_cursor: date_str_cursor + 4]):
                        tz_str = date_str[date_str_cursor: date_str_cursor + 4]
                        date_str_cursor += 4
                    # e.g. EST, EDT, PST
                    elif date_str_cursor + 3 <= date_str_len and is_supported_tz_abbr(date_str[date_str_cursor: date_str_cursor + 3]):
                        tz_str = date_str[date_str_cursor: date_str_cursor + 3]
                        date_str_cursor += 3
                    # e.g. PT, ET, CT
                    elif date_str_cursor + 2 <= date_str_len and is_supported_tz_abbr(date_str[date_str_cursor: date_str_cursor + 2]):
                        tz_str = date_str[date_str_cursor: date_str_cursor + 2]
                        date_str_cursor += 2
                    else:
                        raise ValueError('Bad value for %Z:', date_str)
                elif directive == 'z':
                    # [+|-]DDDD or [+|-]DD:DD, e.g. +0000, +1200
                    if date_str_cursor + 5 <= date_str_len and is_supported_tz_abbr(date_str[date_str_cursor: date_str_cursor + 5]):
                        tz_str = date_str[date_str_cursor: date_str_cursor + 5]
                        date_str_cursor += 5
                    # [+|-]DD or [+|-]DD
                    elif date_str_cursor + 3 <= date_str_len and is_supported_tz_abbr(date_str[date_str_cursor: date_str_cursor + 3]):
                        tz_str = date_str[date_str_cursor: date_str_cursor + 3]
                        date_str_cursor += 3
                    else:
                        raise ValueError('Bad value for %z:', date_str)
                elif directive == 'p':
                    tok = date_str[date_str_cursor: date_str_cursor + 2]
                    if tok != AM and tok != PM:
                        raise ValueError('Bad value for %p:', date_str)
                    ampm = tok
                    date_str_cursor += 2
                else:
                    raise ValueError('Unsupported directive:', directive)

                format_cursor += 1
            else:
                format_cursor += 1
                date_str_cursor += 1

        if ampm != '':
            if hour > 12 or hour < 1:
                raise ValueError('AM/PM specified. hour must be between 1 and 12')
            if ampm == AM and hour == 12:
                hour = 0
            elif ampm == PM:
                hour = hour + 12

        if not is_date(year, month, day):
            raise ValueError('Invalid date:', date_str)
        if not is_time(hour, minute, second):
            raise ValueError('Invalid time:', date_str)

        if month == -1:
            month = 1
        if day == -1:
            day = 1

        datetime = date_to_seconds(year, month, day) + time_to_seconds(hour, minute, second)

        if is_supported_tz_abbr(tz_str):
            datetime += tz_abbr_to_utc_offset(tz_str, datetime)
        else:
            raise ValueError('Unrecognized timezone: ', tz_str)

        return datetime
