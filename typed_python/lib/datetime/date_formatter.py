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

from typed_python import Class, Dict, Entrypoint, Final, ListOf
from typed_python.lib.datetime.date_time import Timezone, UTC

# int to string month mapping where 1 => January
INT_TO_MONTH_NAMES = Dict(int, str)(
    {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
)

# int to string abbreviated month mapping where 1 => Jan
INT_TO_MONTH_ABBR = Dict(int, str)(
    {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
)

# int to string abbreviated day mapping where 0 => Sunday
INT_TO_DAY_NAMES = Dict(int, str)(
    {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
    }
)

# int to string abbreviated day mapping where 0 => Sun
INT_TO_DAY_ABBR = Dict(int, str)(
    {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
)


@Entrypoint
def convert_to_12h(hour: int):
    return (
        12
        if (hour == 0 or hour == 12 or hour == 24)
        else (hour if hour < 12 else hour - 12)
    )


class DateFormatter(Class, Final):
    @Entrypoint
    @staticmethod
    def isoformat(ts: float, timezone: Timezone = UTC) -> str:
        return DateFormatter.format(ts, timezone, "%Y-%m-%dT%H:%M:%S")

    @Entrypoint
    @staticmethod
    def f2d(num: int) -> str:
        """
        Converts a int to string and left pads it to 2 digits
        Parameters:
            num (int): The int to format
        Returns:
            (str): a 2 digit string representation of the int
        """
        res = str(num)
        if len(res) == 1:
            return "0" + res
        return res

    @Entrypoint
    @staticmethod
    def f3d(num: int) -> str:
        """
        Converts a int to string and left pads it to 3 digits
        Parameters:
            num (int): The int to format
        Returns:
            (str): a 3 digit string representation of the int
        """
        res = str(num)
        l = len(res)
        if l == 2:
            return "0" + res
        elif l == 1:
            return "00" + res
        return res

    @Entrypoint
    @staticmethod
    def f4d(num: int) -> str:
        """
        C converts a int to string and left pads it with zeroes to 4 digits
        Parameters:
            num (int): The int to format
        Returns:
            (str): a 4 digit string representation of the int
        """
        res = str(num)
        l = len(res)
        if l == 3:
            return "0" + res
        elif l == 2:
            return "00" + res
        elif l == 1:
            return "000" + res
        return res

    @Entrypoint
    @staticmethod
    def format(
        ts: float = 0, timezone: Timezone = UTC, format: str = "%Y-%m-%d %H:%M:%S"
    ) -> str:
        """
        Converts a Timestamp to a string in a given format
        Parameters:
            utc_offset (int): The offset from UTC in seconds
            format (str): A string specifying formatting directives. E.g. '%Y-%m-%d %H:%M:%S'
        Returns:
            date_str(str): A string representing the date in the specified format. E.g. "Mon January 2, 2021"
        """
        datetime = timezone.datetime(ts)
        weekday = datetime.date.weekday()
        doy = datetime.date.dayOfYear()
        y = datetime.date.year
        m = datetime.date.month
        d = datetime.date.day

        h = datetime.timeOfDay.hour
        min = datetime.timeOfDay.minute
        s = datetime.timeOfDay.second

        # short circuits for common formats
        if format == "%Y-%m-%d":
            return "-".join(
                ListOf(str)(
                    [DateFormatter.f4d(y), DateFormatter.f2d(m), DateFormatter.f2d(d)]
                )
            )

        if format == "%Y-%m-%d %H:%M:%S":
            return "".join(
                ListOf(str)(
                    [
                        DateFormatter.f4d(y),
                        "-",
                        DateFormatter.f2d(m),
                        "-",
                        DateFormatter.f2d(d),
                        " ",
                        DateFormatter.f2d(h),
                        ":",
                        DateFormatter.f2d(min),
                        ":",
                        DateFormatter.f2d(s),
                    ]
                )
            )

        result = ListOf(str)()

        pos = 0
        strlen = len(format)

        while pos < strlen:
            if format[pos] == "%" and pos + 1 < strlen:
                directive = format[pos + 1]

                if directive == "Y":
                    result.append(DateFormatter.f4d(y))
                elif directive == "m":
                    result.append(DateFormatter.f2d(m))
                elif directive == "d":
                    result.append(DateFormatter.f2d(d))
                elif directive == "H":
                    result.append(DateFormatter.f2d(h))
                elif directive == "M":
                    result.append(DateFormatter.f2d(min))
                elif directive == "S":
                    result.append(DateFormatter.f2d(s))
                elif directive == "a":
                    result.append(INT_TO_DAY_ABBR[weekday])
                elif directive == "A":
                    result.append(INT_TO_DAY_NAMES[weekday])
                elif directive == "w":
                    result.append(str(weekday))
                elif directive == "b":
                    result.append(INT_TO_MONTH_ABBR[m])
                elif directive == "B":
                    result.append(INT_TO_MONTH_NAMES[m])
                elif directive == "y":
                    result.append(DateFormatter.f2d(y % 100))
                elif directive == "I":
                    result.append(DateFormatter.f2d(convert_to_12h(h)))
                elif directive == "p":
                    result.append("AM" if h < 12 else "PM")
                elif directive == "Z":
                    result.append(
                        "UTC"
                    )  # timestamps don't store tz data, are pegged to UTC
                elif directive == "z":
                    result.append(
                        "+0000"
                    )  # timestamps don't store tz data, are pegged to UTC
                elif directive == "j":
                    result.append(DateFormatter.f3d(doy))  # day number of year
                elif directive == "C":
                    result.append(DateFormatter.f2d(y // 100))  # century
                elif directive == "%":
                    result.append("%")
                elif directive == "u":
                    result.append(
                        str(7 if weekday == 0 else weekday)
                    )  # ISO weekday 1-7
                else:
                    raise ValueError("Unsupported formatting directive: " + directive)
                pos += 1
            else:
                result.append(format[pos])
            pos += 1
        return "".join(result)
