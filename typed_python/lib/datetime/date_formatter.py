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

from typed_python import Class, Final, Entrypoint, ListOf, Dict
from typed_python.lib.datetime.chrono import is_leap_year, convert_to_12h

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
                    result.append(DateFormatter.f2d(convert_to_12h(h)))
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
