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

from typed_python import Class, NamedTuple, Final
from typed_python import Entrypoint, ListOf, Dict
from typed_python.lib.datetime.timezone_offset import utc_offset_by_tz_abbreviation, is_tz_offset
from typed_python.lib.datetime.chrono import time_to_seconds, date_to_seconds, is_leap_year, is_date, is_time


Date = NamedTuple(year=int, month=int, day=int, hour=int, minute=int, second=int, ms=int, us=int, ns=int, weekday=int, doy=int)

months = Dict(str, int)({'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3, 'apr': 4,
                         'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
                         'sep': 9, 'sept': 9, 'september': 9, 'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
                         'dec': 12, 'december': 12})


class DateParser(Class, Final):

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
        return month_str.strip().lower() in months

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
        return year_str.isdigit() and len(year_str) == 4

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
        while cursor < len(time_str):
            token = ''
            while cursor < len(time_str) and time_str[cursor].isalpha():
                token += time_str[cursor]
                cursor += 1

            if token != '':
                tokens.append(token)

            token = ''
            while cursor < len(time_str) and (time_str[cursor].isdigit() or time_str[cursor] == '.'):
                token += time_str[cursor]
                cursor += 1
                continue

            if token != '':
                tokens.append(token)

            token = ''
            while cursor < len(time_str) and time_str[cursor].isspace():
                token += time_str[cursor]
                cursor += 1

            if token != '':
                if skip_chars.find(' ') > -1:
                    pass
                else:
                    tokens.append(' ')

            token = ''
            while (cursor < len(time_str) and not time_str[cursor].isspace()
                   and not time_str[cursor].isdigit() and not time_str[cursor].isalpha()):
                token += time_str[cursor]
                cursor += 1

            if token != '':
                if skip_chars.find(token) > -1:
                    pass
                else:
                    tokens.append(token)
        return tokens

    @Entrypoint
    @staticmethod
    def parse_tz_offset(tokens: ListOf(str)) -> int:
        '''
        Converts a set of tokens representing a timezone offset to seconds.
        Parameters:
            tokens (ListOf(str)): A set of string tokens representing a timezone. E.g. ['Z'] or ['+', '02', ':', '23']
            Returns:
            The offset in seconds
        '''
        if tokens[0] != '+' and tokens[0] != '-':
            raise ValueError("tz offset must begin with '+' or '-'")

        hour = min = 0
        second = 0.0
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

        hour = hour * -1 if tokens[0] == '-' else hour

        if is_tz_offset(hour, min, second):
            return hour * 3600 + (min * 60 if hour > 0 else min * -60) + (second if hour > 0 else second * -1)
        else:
            raise ValueError('Invalid tz offset: ')

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
        tokens = DateParser.get_tokens(time_str=date_str, skip_chars='/-:')

        # Process date segment
        segment = ListOf(str)([])
        cursor = 0
        while cursor < len(tokens):
            if tokens[cursor] == 'T' or tokens[cursor] == 't' or tokens[cursor] == ' ':
                cursor += 1
                break
            elif tokens[cursor] == '+' or tokens[cursor] == '-':
                cursor += 1
                break
            else:
                segment.append(tokens[cursor])
                cursor += 1

        year = month = day = -1

        if len(segment) == 1:
            if len(segment[0]) == 8:
                year, month, day = int(segment[0][:4]), int(segment[0][4:6]), int(segment[0][6:8])
            elif len(segment[0]) == 6:
                year, month, day = int(segment[0][:4]), int(segment[0][4:6]), 1
            elif len(segment[0]) == 4:
                year, month, day = int(segment[0][:4]), 1, 1
        elif len(segment) == 2:
            year, month, day = int(segment[0]), int(segment[1]), 1
        elif len(segment) == 3:
            year, month, day = int(segment[0]), int(segment[1]), int(segment[2])

        if not is_date(year, month, day):
            raise ValueError('Invalid date: ', segment)

        # Process time segement
        segment.clear()
        while cursor < len(tokens):
            if tokens[cursor] == 'T' or tokens[cursor] == 't' or tokens[cursor] == ' ':
                cursor += 1
                break
            elif tokens[cursor] == '+' or tokens[cursor] == '-' or tokens[cursor].isalpha():
                break
            else:
                segment.append(tokens[cursor])
                cursor += 1

        hour = minute = 0
        second = 0.0

        if len(segment) == 1:
            if len(segment[0]) == 6:
                hour, minute, second = int(segment[0][:2]), int(segment[0][2:4]), float(segment[0][4:6])
            elif len(segment[0]) == 4:
                hour, minute, second = int(segment[0][:2]), int(segment[0][2:4]), 0.0
            elif len(segment[0]) == 2:
                hour, minute, second = int(segment[0][:2]), 0, 0.0
        elif len(segment) == 2:
            hour, minute, second = int(segment[0]), int(segment[1]), 0.0
        elif len(segment) == 3:
            hour, minute, second = int(segment[0]), int(segment[1]), float(segment[2])

        if not is_time(hour, minute, second):
            raise ValueError('Invalid time: ', segment)

        # Process timezone segment
        segment.clear()
        while cursor < len(tokens):
            segment.append(tokens[cursor])
            cursor += 1

        dt = date_to_seconds(year, month, day) + time_to_seconds(hour, minute, second)
        if len(segment) == 0:
            tz_offset = 0
        elif len(segment) == 1:
            tz_offset = utc_offset_by_tz_abbreviation(segment[0], dt)
        elif segment[0] == '+' or segment[0] == '-':
            tz_offset = DateParser.parse_tz_offset(segment)
        else:
            raise ValueError('Unsupported tz format', segment)

        return dt + tz_offset

    @Entrypoint
    @staticmethod
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
        sep_idx = None
        for idx, token in enumerate(tokens):
            if token == 'am' or token == 'pm':
                sep_idx = idx
                break

        if sep_idx is not None:
            time_part, ampm_part = tokens[:sep_idx], tokens[sep_idx:]
        else:
            time_part = tokens
            ampm_part = None

        if len(time_part) == 5 and time_part[1] == ':' and time_part[2].isdigit() and time_part[3] == ':':
            # HH:MM:SS
            if time_part[0].isdigit() and time_part[4].isdigit():
                h, m, s = int(time_part[0]), int(time_part[2]), int(time_part[4])
            else:
                raise ValueError()
        elif len(time_part) == 3:
            # HH:MM
            if time_part[0].isdigit() and time_part[1] == ':' and time_part[2].isdigit():
                h, m, s = int(time_part[0]), int(time_part[2]), 0
            else:
                raise ValueError()
        else:
            raise ValueError('Unsupported time format', tokens)

        if ampm_part is not None:
            if h > 12 or h < 1:
                raise ValueError('AM/PM specified. hour must be between 1 and 12')
            if ampm_part[0].lower() == 'am' and h == 12:
                h = 0
            elif ampm_part[0].lower() == 'pm':
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
        date_str = date_str.lower().replace('a.m.', 'am').replace('p.m.', 'pm').strip()
        tokens = DateParser.get_tokens(date_str, skip_chars=' ,')

        # if/elsif block is long but it's simple and allows us to clearly define the formats we support
        # and add new formats as needed

        y = m = d = 0
        time_tokens = None

        # 5+ tokens with 4 digit year as 5th token
        if len(tokens) >= 5 and DateParser.is_year(tokens[4]):
            # DD/Month/YYYY or DD-Month-YYYY
            if (DateParser.is_month(tokens[2]) and ((tokens[1] == '/' and tokens[3] == '/') or
               (tokens[1] == '-' and tokens[3] == '-'))):
                y, m, d, time_tokens = int(tokens[4]), months[tokens[2].lower()], int(tokens[0]), tokens[5:]

            # Month-DD-YYYY
            elif DateParser.is_month(tokens[0]) and tokens[1] == '-' and tokens[2].isdigit() and tokens[3] == '-':
                y, m, d, time_tokens = int(tokens[4]), months[tokens[0].lower()], int(tokens[2]), tokens[5:]

            # Month-DD-YYYY or Month/DD/YYYY
            elif (DateParser.is_month(tokens[0]) and ((tokens[1] == '/' and tokens[3] == '/') or
                                                      (tokens[1] == '-' and tokens[3] == '-'))):
                y, m, d, time_tokens = int(tokens[4]), months[tokens[0].lower()], int(tokens[2]), tokens[5:]

            else:
                raise ValueError('Unsupported date format: ' + date_str)

        # 5+ tokens with 4 digit year as 1st token
        elif len(tokens) >= 5 and DateParser.is_year(tokens[0]) and DateParser.is_month(tokens[2]) and tokens[4].isdigit():
            # YYYY/Month/DD or YYYY-Month-DD
            if (tokens[1] == '/' and tokens[3] == '/') or (tokens[1] == '-' and tokens[3] == '-'):
                y, m, d, time_tokens = int(tokens[0]), months[tokens[2].lower()], int(tokens[4]), tokens[5:]
            else:
                raise ValueError('Unsupported date format: ' + date_str)

        # Month D YYYY
        elif len(tokens) >= 3 and DateParser.is_month(tokens[0]) and tokens[1].isdigit() and DateParser.is_year(tokens[2]):
            y, m, d, time_tokens = int(tokens[2]), months[tokens[0].lower()], int(tokens[1]), tokens[3:]

        # D Month YYYY
        elif len(tokens) >= 3 and DateParser.is_month(tokens[1]) and tokens[0].isdigit() and DateParser.is_year(tokens[2]):
            y, m, d, time_tokens = int(tokens[2]), months[tokens[1].lower()], int(tokens[0]), tokens[3:]

        # YYYY Month DD
        elif len(tokens) >= 3 and DateParser.is_year(tokens[0]) and DateParser.is_month(tokens[1]) and tokens[2].isdigit():
            y, m, d, time_tokens = int(tokens[0]), months[tokens[1].lower()], int(tokens[2]), tokens[3:]

        else:
            raise ValueError('Unsupported date format: ' + date_str)

        if not is_date(y, m, d):
            raise ValueError('Invalid date: ' + date_str)

        return date_to_seconds(y, m, d) + DateParser.parse_non_iso_time(time_tokens)

    @Entrypoint
    @staticmethod
    def utc_offset_string_to_seconds(utc_offset: str) -> int:
        '''
        Converts a tz offset in the form [+|-]HH[:]MM to seconds
        Parameters:
            utc_offset(string): The utc offset
        Returns:
            The utc offset in seconds
        '''
        offset = ''.join(utc_offset.split(':'))
        hrs = int(offset[0:3])
        mins = int(offset[3:5])

        if is_tz_offset(hrs, mins):
            return hrs * 3600 + (mins * 60 if hrs > 0 else mins * -60)
        else:
            raise ValueError('Invalid tz offset: ' + utc_offset)


# int to string for DateFormatter
month_names = Dict(int, str)({
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December',
})

month_abbr = Dict(int, str)({
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec',
})

day_names = Dict(int, str)({
    0: 'Sunday',
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday'
})

day_abbr = Dict(int, str)({
    0: 'Sun',
    1: 'Mon',
    2: 'Tue',
    3: 'Wed',
    4: 'Thu',
    5: 'Fri',
    6: 'Sat'
})


class DateFormatter(Class, Final):

    @Entrypoint
    @staticmethod
    def convert_to_12h(hour: int):
        if hour == 0 or hour == 12 or hour == 24:
            return 12
        elif hour < 12:
            return hour
        else:
            return hour - 12

    @Entrypoint
    @staticmethod
    def isoformat(ts: float, utc_offset: int = 0):
        return DateFormatter.format(ts=ts, format='%Y-%m-%dT%H:%M:%S', utc_offset=utc_offset)

    @Entrypoint
    @staticmethod
    def f2d(num: int) -> str:
        '''
        Converts a int to string and left pads it to 2 digits
        Parameters:
            num (int): The int to format
        Returns:
            (str): a 2 digit string representation of the int
        '''
        res = str(num)
        if len(res) == 1:
            return '0' + res
        return res

    @Entrypoint
    @staticmethod
    def f3d(num: int) -> str:
        '''
        Converts a int to string and left pads it to 3 digits
        Parameters:
            num (int): The int to format
        Returns:
            (str): a 3 digit string representation of the int
        '''
        res = str(num)
        if len(res) == 2:
            return '0' + res
        elif len(res) == 1:
            return '00' + res
        return res

    @Entrypoint
    @staticmethod
    def f4d(num: int) -> str:
        '''
        C converts a int to string and left pads it with zeroes to 4 digits
        Parameters:
            num (int): The int to format
        Returns:
            (str): a 4 digit string representation of the int
        '''
        res = str(num)
        if len(res) == 3:
            return '0' + res
        elif len(res) == 2:
            return '00' + res
        elif len(res) == 1:
            return '000' + res
        return res

    @ Entrypoint
    @ staticmethod
    def format(ts: float = 0, utc_offset: int = 0, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        '''
        Converts a Timestamp to a string in a given format
        Parameters:
            utc_offset (int): The offset from UTC in seconds
            format (str): A string specifying formatting directives. E.g. '%Y-%m-%d %H:%M:%S'
        Returns:
            date_str(str): A string representing the date in the specified format. E.g. "Mon January 2, 2021"
        '''

        ts = ts + utc_offset
        z = ts // 86400 + 719468
        era = (z if z >= 0 else z - 146096) // 146097
        doe = z - era * 146097
        yoe = (doe - (doe // 1460) + (doe // 36524) - (doe // 146096)) // 365
        y = int(yoe + era * 400)
        doy = int(doe - ((365 * yoe) + (yoe // 4) - (yoe // 100)))
        mp = (5 * doy + 2) // 153
        d = int(doy - (153 * mp + 2) // 5 + 1)
        m = int(mp + (3 if mp < 10 else -9))
        y += (m <= 2)

        h = int((ts // 3600) % 24)
        min = int((ts // (3600 / 60)) % 60)
        s = (ts // (3600 / 60 / 60)) % (60)

        # http://howardhinnant.github.io/date_algorithms.html#weekday_from_days
        days = int(ts // 86400)
        weekday = int((days + 4) % 7 if days >= -4 else (days + 5) % 7 + 6)

        # The above algorithm is based on a year starting on March 1.
        # We'll have to shift this to January 1 based year by adding 60 days and wrapping at
        # at year end
        doy += 60
        doy = doy % 365 if doy > 365 else doy

        if is_leap_year(y) and m > 2:
            doy += 1

        result = ListOf(str)([])

        pos = 0

        strlen = len(format)

        while pos < strlen:
            if format[pos] == '%' and pos + 1 < strlen:
                directive = format[pos + 1]

                if directive == 'a':
                    result.append(day_abbr[weekday])
                    pos += 1
                elif directive == 'A':
                    result.append(day_names[weekday])
                    pos += 1
                elif directive == 'w':
                    result.append(str(weekday))
                    pos += 1
                elif directive == 'd':
                    result.append(DateFormatter.f2d(d))
                    pos += 1
                elif directive == 'b':
                    result.append(month_abbr[m])
                    pos += 1
                elif directive == 'B':
                    result.append(month_names[m])
                    pos += 1
                elif directive == 'm':
                    result.append(DateFormatter.f2d(m))
                    pos += 1
                elif directive == 'y':
                    result.append(DateFormatter.f2d(y % 100))
                    pos += 1
                elif directive == 'Y':
                    result.append(DateFormatter.f4d(y))
                    pos += 1
                elif directive == 'H':
                    result.append(DateFormatter.f2d(h))
                    pos += 1
                elif directive == 'I':
                    result.append(DateFormatter.f2d(DateFormatter.convert_to_12h(h)))
                    pos += 1
                elif directive == 'p':
                    result.append('AM' if h < 12 else 'PM')
                    pos += 1
                elif directive == 'M':
                    result.append(DateFormatter.f2d(min))
                    pos += 1
                elif directive == 'S':
                    result.append(DateFormatter.f2d(s))
                    pos += 1
                elif directive == 'Z':
                    result.append('UTC')  # timestamps don't store tz data, are pegged to UTC
                    pos += 1
                elif directive == 'z':
                    result.append('+0000')  # timestamps don't store tz data, are pegged to UTC
                    pos += 1
                elif directive == 'j':
                    result.append(DateFormatter.f3d(doy))  # day number of year
                    pos += 1
                elif directive == 'C':
                    result.append(DateFormatter.f2d(y // 100))  # century
                    pos += 1
                elif directive == '%':
                    result.append('%')
                    pos += 1
                elif directive == 'u':
                    result.append(str(7 if weekday == 0 else weekday))  # ISO weekday 1-7
                    pos += 1
                else:
                    result.append(directive)
                    pos += 1
            else:
                result.append(format[pos])
            pos += 1
        return ''.join(result)
