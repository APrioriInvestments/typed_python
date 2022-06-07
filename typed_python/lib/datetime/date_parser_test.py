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

import unittest
from typed_python.lib.datetime.date_parser import DateParser
import pytest
from datetime import datetime, timedelta
import pytz


def get_years_in_range(start, end):
    dates = []
    for i in range(start, end):
        dates.append(datetime(i, 1, 1, 0, 0, 0, 0, pytz.UTC))

    return dates


def get_months_in_year(year):
    dates = []
    for i in range(1, 13):
        dates.append(datetime(year, i, 1, 0, 0, 0, 0, pytz.UTC))

    return dates


def get_datetimes_in_range(start, end, step):
    d = start
    dates = [start]

    while d < end:
        if step == 'days':
            d += timedelta(days=1)
        elif step == 'hours':
            d += timedelta(hours=1)
        elif step == 'minutes':
            d += timedelta(minutes=1)
        elif step == 'seconds':
            d += timedelta(seconds=1)
        else:
            raise ValueError('Unsupported step: ' + step)
        dates.append(d)
    return dates


def time_to_sec(hours=0, mins=0, secs=0, fract=0):
    return (hours * 60 * 60) + (mins * 60) + secs + fract


class TestDateParser(unittest.TestCase):

    def test_empty_string(self):
        with pytest.raises(ValueError):
            DateParser.parse('')

    def test_fails_on_random_text(self):
        with pytest.raises(ValueError):
            DateParser.parse('scuse me while i kiss the sky')

    def test_fails_with_extra_text(self):
        with pytest.raises(ValueError):
            DateParser.parse('1997-01-01 and some more text')

    def test_parse_invalid_year(self):
        days = [
            'a997',  # not 4 digit number
            '97',
        ]
        for day in days:
            with pytest.raises(ValueError):
                DateParser.parse(day)

    def test_parse_valid_year(self):
        days = [
            '1997',
            '2020',
            '9999',
            '0000'
        ]
        for day in days:
            DateParser.parse_iso(day)

    def test_parse_invalid_month(self):
        days = [
            '1997-00',
            '1997-13',
            '1997-ab'
        ]
        for day in days:
            with pytest.raises(ValueError):
                DateParser.parse_iso(day)

    def test_parse_invalid_day(self):
        days = [
            '1997-01-00',  # day < 1
            '1997-01-32',  # day > 31
            '1997-04-31',  # day > 30 in Apr
            '1997-06-31',  # day > 30 in June
            '1997-09-31',  # day > 30 in Sep
            '1997-11-31',  # day > 30 in Nov
            '1997-02-29',  # day > 28 for non-leap year Feb
            '2020-02-30',  # day > 30 for leap year Feb
            '2020-02-ab',  # day is not digit
            '1900-02-29',  # year is multiple of 4, but not leap year so no 29
        ]

        for day in days:
            with pytest.raises(ValueError):
                DateParser.parse_iso(day)

    def test_parse_iso_yyyyx(self):
        years = get_years_in_range(1942, 1970) + get_years_in_range(2001, 2022)
        for year in years:
            assert DateParser.parse_iso(year.strftime('%Y')) == datetime.timestamp(year), year.strftime('%Y')

    def test_parse_iso_yyyymm(self):
        months = get_months_in_year(1999) + get_months_in_year(2020)
        formats = [
            '%Y-%m',
            '%Y/%m',
            '%Y%m',
        ]
        for format in formats:
            for month in months:
                assert DateParser.parse(month.strftime(format)) == datetime.timestamp(month), month.strftime(format)

    def test_parse_iso_yyyymmdd(self):
        # all days in non leap year and leap year
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y%m%d',

        ]
        for format in formats:
            for day in days:
                assert DateParser.parse_iso(day.strftime(format)) == datetime.timestamp(day), day.strftime(format)

    def test_parse_iso_yyyymmddhh(self):
        # all hours in feb 2020
        hours = get_datetimes_in_range(start=datetime(2020, 2, 1, 0, 0, 0, 0, pytz.UTC),
                                       end=datetime(2020, 3, 2, 0, 0, 0, 0, pytz.UTC),
                                       step='hours')
        formats = [
            '%Y-%m-%dT%H',
            '%Y-%m-%dT%HZ',
            '%Y-%m-%d %H',
            '%Y/%m/%dT%H',
            '%Y/%m/%d %H',
            '%Y%m%dT%H',
            '%Y%m%d %H',
            '%Y-%m-%dT%HZ',
            '%Y-%m-%dT%H+00',
            '%Y-%m-%dT%H+00:00'
        ]

        for format in formats:
            for hour in hours:
                assert DateParser.parse_iso(hour.strftime(format)) == datetime.timestamp(hour), hour.strftime(format)

    def test_parse_iso_yyyymmddhhmm(self):
        minutes = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 0, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 15, 0, 0, 0, pytz.UTC),
                                         step='minutes')
        formats = [
            '%Y-%m-%dT%H:%M',
            '%Y-%m-%dT%H:%MZ',
            '%Y-%m-%d %H:%M',
            '%Y/%m/%dT%H:%M',
            '%Y/%m/%d %H:%M',
            '%Y%m%dT%H:%M',
            '%Y%m%d %H:%M',
            '%Y-%m-%dT%H:%MZ',
            '%Y-%m-%dT%H:%M+00',
            '%Y-%m-%dT%H:%M+00:00'
        ]

        for format in formats:
            for minute in minutes:
                assert DateParser.parse(minute.strftime(format)) == datetime.timestamp(minute), minute.strftime(format)

    def test_parse_iso_yyyymmddhhmmss(self):
        seconds = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
                                         step='seconds')
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%dT%H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y%m%dT%H:%M:%S',
            '%Y%m%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S+00',
            '%Y-%m-%dT%H:%M:%S+00:00'
        ]

        for format in formats:
            for second in seconds:
                assert DateParser.parse_iso(second.strftime(format)) == datetime.timestamp(second), second.strftime(format)

    def test_parse_iso_yyyymmddhhmmsssss(self):
        seconds = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
                                         step='seconds')
        formats = [
            '%Y-%m-%dT%H:%M:%S.123',
            '%Y-%m-%dT%H:%M:%S.123Z',
            '%Y-%m-%d %H:%M:%S.123',
            '%Y/%m/%dT%H:%M:%S.123',
            '%Y/%m/%d %H:%M:%S.123',
            '%Y%m%dT%H:%M:%S.123',
            '%Y%m%d %H:%M:%S.123',
            '%Y-%m-%dT%H:%M:%S.123Z',
            '%Y-%m-%dT%H:%M:%S.123+00',
            '%Y-%m-%dT%H:%M:%S.123+00:00'
        ]

        for format in formats:
            for second in seconds:
                assert DateParser.parse_iso(second.strftime(format)) == datetime.timestamp(second) + .123, second.strftime(format)

    def test_parse_iso_with_tz_offset(self):
        hours = get_datetimes_in_range(start=datetime(2020, 2, 1, 0, 0, 0, 0, pytz.UTC),
                                       end=datetime(2020, 3, 2, 0, 0, 0, 0, pytz.UTC),
                                       step='hours')

        tz_offset = 4500

        formats = [
            '%Y-%m-%dT%H:%M+01:15',
            '%Y-%m-%d %H:%M+01:15',
        ]

        for format in formats:
            for hour in hours:
                assert DateParser.parse_iso(hour.strftime(format)) == datetime.timestamp(hour) + tz_offset, hour.strftime(format)

    def test_parse_non_iso_with_whitespace(self):
        hours = get_datetimes_in_range(start=datetime(2020, 2, 1, 0, 0, 0, 0, pytz.UTC),
                                       end=datetime(2020, 3, 2, 0, 0, 0, 0, pytz.UTC),
                                       step='hours')
        formats = [
            ' %B-%d-%Y %H:%M',
            '%B-%d-%Y %H:%M ',
            ' %B-%d-%Y %H:%M ',
        ]

        for format in formats:
            for hour in hours:
                assert DateParser.parse(hour.strftime(format)) == datetime.timestamp(hour), hour.strftime(format)

    def test_parse_non_iso_dates(self):
        # all days in non leap year and leap year
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')
        supported_formats = [
            '%b %d %Y',   # e.g Jan 1 1997
            '%B %d %Y',   # e.g January 1 1997
            '%b %d, %Y',  # e.g Jan 1, 1997
            '%B %d, %Y',  # e.g January 1, 1997
            '%b%d, %Y',   # e.g Jan1, 1997
            '%B%d, %Y',   # e.g January1, 1997
            '%b %d,%Y',   # e.g Jan1, 1997
            '%B %d,%Y',   # e.g January1, 1997
            '%d %b %Y',   # e.g 1 Jan 1997
            '%d %B %Y',   # e.g 1January 1997
            '%d%b%Y',     # e.g 1Jan1997
            '%d%B%Y',     # e.g 1January1997
            '%d/%b/%Y',   # e.g 1/Jan/1997
            '%d/%B/%Y',   # e.g 1/January/1997
            '%d-%b-%Y',   # e.g 1-Jan-1997
            '%d-%B-%Y',   # e.g 1-January-1997
            '%Y %b %d',   # e.g 1997 Jan 1
            '%Y %B %d',   # e.g 1997 January 1
            '%Y/%b/%d',   # e.g 1997/Jan/1
            '%Y/%B/%d',   # e.g 1997/January/1
            '%Y-%b-%d',   # e.g 1997-Jan-1
            '%Y-%B-%d',   # e.g 1997-January-1
            '%b-%d-%Y',   # e.g Jan-1-1997
            '%B-%d-%Y',   # e.g January-1-1997
            '%b/%d/%Y',   # e.g Jan/1/1997
            '%B/%d/%Y',   # e.g January/1/1997
        ]
        for format in supported_formats:
            for day in days:
                assert DateParser.parse_non_iso(day.strftime(format)) == datetime.timestamp(day), day.strftime(format)

    def test_parse_non_iso_yyyymmddhhmm(self):
        minutes = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 0, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 15, 0, 0, 0, pytz.UTC),
                                         step='minutes')
        supported_formats = [
            '%b %d %Y %H:%M',   # e.g Jan 1 1997 13:50
            '%B %d %Y %H:%M',   # e.g January 1 1997 13:50
            '%b %d, %Y %H:%M',  # e.g Jan 1, 1997 13:50
            '%B %d, %Y %H:%M',  # e.g January 1, 1997 13:50
            '%b%d, %Y %H:%M',   # e.g Jan1, 1997 13:50
            '%B%d, %Y %H:%M',   # e.g January1, 1997 13:50
            '%b %d,%Y %H:%M',   # e.g Jan1, 1997 13:50
            '%B %d,%Y %H:%M',   # e.g January1, 1997 13:50
            '%d %b %Y %H:%M',   # e.g 1 Jan 1997 13:50
            '%d %B %Y %H:%M',   # e.g 1January 1997 13:50
            '%d%b%Y %H:%M',     # e.g 1Jan1997 13:50
            '%d%B%Y %H:%M',     # e.g 1January1997 13:50
            '%d/%b/%Y %H:%M',   # e.g 1/Jan/1997 13:50
            '%d/%B/%Y %H:%M',   # e.g 1/January/1997 13:50
            '%d-%b-%Y %H:%M',   # e.g 1-Jan-1997 13:50
            '%d-%B-%Y %H:%M',   # e.g 1-January-1997 13:50
            '%Y %b %d %H:%M',   # e.g 1997 Jan 1 13:50
            '%Y %B %d %H:%M',   # e.g 1997 January 1 13:50
            '%Y/%b/%d %H:%M',   # e.g 1997/Jan/1 13:50
            '%Y/%B/%d %H:%M',   # e.g 1997/January/1 13:50
            '%Y-%b-%d %H:%M',   # e.g 1997-Jan-1 13:50
            '%Y-%B-%d %H:%M',   # e.g 1997-January-1 13:50
            '%b-%d-%Y %H:%M',   # e.g Jan-1-1997 13:50
            '%B-%d-%Y %H:%M',   # e.g January-1-1997 13:50
            '%b/%d/%Y %H:%M',   # e.g Jan/1/1997 13:50
            '%B/%d/%Y %H:%M',   # e.g January/1/1997 13:50
        ]
        for format in supported_formats:
            for minute in minutes:
                assert DateParser.parse_non_iso(minute.strftime(format)) == datetime.timestamp(minute), minute.strftime(format)

    def test_parse_non_iso_yyyymmddhhmmss(self):
        seconds = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
                                         step='seconds')
        supported_formats = [
            '%b %d %Y %H:%M:%S',   # e.g Jan 1 1997 13:50
            '%B %d %Y %H:%M:%S',   # e.g January 1 1997 13:50
            '%b %d, %Y %H:%M:%S',  # e.g Jan 1, 1997 13:50
            '%B %d, %Y %H:%M:%S',  # e.g January 1, 1997 13:50
            '%b%d, %Y %H:%M:%S',   # e.g Jan1, 1997 13:50
            '%B%d, %Y %H:%M:%S',   # e.g January1, 1997 13:50
            '%b %d,%Y %H:%M:%S',   # e.g Jan1, 1997 13:50
            '%B %d,%Y %H:%M:%S',   # e.g January1, 1997 13:50
            '%d %b %Y %H:%M:%S',   # e.g 1 Jan 1997 13:50
            '%d %B %Y %H:%M:%S',   # e.g 1January 1997 13:50
            '%d%b%Y %H:%M:%S',     # e.g 1Jan1997 13:50
            '%d%B%Y %H:%M:%S',     # e.g 1January1997 13:50
            '%d/%b/%Y %H:%M:%S',   # e.g 1/Jan/1997 13:50
            '%d/%B/%Y %H:%M:%S',   # e.g 1/January/1997 13:50
            '%d-%b-%Y %H:%M:%S',   # e.g 1-Jan-1997 13:50
            '%d-%B-%Y %H:%M:%S',   # e.g 1-January-1997 13:50
            '%Y %b %d %H:%M:%S',   # e.g 1997 Jan 1 13:50
            '%Y %B %d %H:%M:%S',   # e.g 1997 January 1 13:50
            '%Y/%b/%d %H:%M:%S',   # e.g 1997/Jan/1 13:50
            '%Y/%B/%d %H:%M:%S',   # e.g 1997/January/1 13:50
            '%Y-%b-%d %H:%M:%S',   # e.g 1997-Jan-1 13:50
            '%Y-%B-%d %H:%M:%S',   # e.g 1997-January-1 13:50
            '%b-%d-%Y %H:%M:%S',   # e.g Jan-1-1997 13:50
            '%B-%d-%Y %H:%M:%S',   # e.g January-1-1997 13:50
            '%b/%d/%Y %H:%M:%S',   # e.g Jan/1/1997 13:50
            '%B/%d/%Y %H:%M:%S',   # e.g January/1/1997 13:50
        ]
        for format in supported_formats:
            for second in seconds:
                assert DateParser.parse_non_iso(second.strftime(format)) == datetime.timestamp(second), second.strftime(format)

    def test_non_iso_pm_indicator(self):

        times = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 0, 0, 0, pytz.UTC),
                                       end=datetime(2020, 2, 29, 23, 59, 0, 0, pytz.UTC),
                                       step='minutes')
        supported_formats = [
            '%B/%d/%Y %I:%MPM',
            '%B/%d/%Y %I:%Mpm',
            '%B/%d/%Y %I:%M:%SPM',
            '%B/%d/%Y %I:%M:%Spm',
        ]

        for format in supported_formats:
            for time in times:
                assert DateParser.parse_non_iso(time.strftime(format)) == datetime.timestamp(time), time.strftime(format)

    def test_non_iso_am_indicator(self):

        times = get_datetimes_in_range(start=datetime(2020, 2, 29, 0, 0, 0, 0, pytz.UTC),
                                       end=datetime(2020, 2, 29, 11, 59, 0, 0, pytz.UTC),
                                       step='minutes')
        supported_formats = [
            '%B/%d/%Y %I:%MAM',
            '%B/%d/%Y %I:%Mam',
            '%B/%d/%Y %I:%M:%SAM',
            '%B/%d/%Y %I:%M:%Sam',
        ]

        for format in supported_formats:
            for time in times:
                assert DateParser.parse_non_iso(time.strftime(format)) == datetime.timestamp(time), time.strftime(format)

    def test_parse_non_iso_invalid_day(self):
        with pytest.raises(ValueError):
            DateParser.parse_non_iso('1997 Jan 32')
        with pytest.raises(ValueError):
            DateParser.parse_non_iso('1997 Jan 0')

    def test_parse_non_iso_invalid_month(self):
        with pytest.raises(ValueError):
            DateParser.parse_non_iso('Janeary 01 1997')

    def test_parse(self):
        # test main entry point with mix of iso and non iso format dates
        seconds = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
                                         step='seconds')
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%dT%H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y%m%dT%H:%M:%S',
            '%Y%m%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S+00',
            '%Y-%m-%dT%H:%M:%S+00:00',
            '%b %d %Y %H:%M:%S',   # e.g Jan 1 1997 13:50
            '%B %d %Y %H:%M:%S',   # e.g January 1 1997 13:50
            '%b %d, %Y %H:%M:%S',  # e.g Jan 1, 1997 13:50
            '%B %d, %Y %H:%M:%S',  # e.g January 1, 1997 13:50
            '%b%d, %Y %H:%M:%S',   # e.g Jan1, 1997 13:50
            '%B%d, %Y %H:%M:%S',   # e.g January1, 1997 13:50
            '%b %d,%Y %H:%M:%S',   # e.g Jan1, 1997 13:50
            '%B %d,%Y %H:%M:%S',   # e.g January1, 1997 13:50
            '%d %b %Y %H:%M:%S',   # e.g 1 Jan 1997 13:50
            '%d %B %Y %H:%M:%S',   # e.g 1January 1997 13:50
            '%d%b%Y %H:%M:%S',     # e.g 1Jan1997 13:50
            '%d%B%Y %H:%M:%S',     # e.g 1January1997 13:50
            '%d/%b/%Y %H:%M:%S',   # e.g 1/Jan/1997 13:50
            '%d/%B/%Y %H:%M:%S',   # e.g 1/January/1997 13:50
            '%d-%b-%Y %H:%M:%S',   # e.g 1-Jan-1997 13:50
            '%d-%B-%Y %H:%M:%S',   # e.g 1-January-1997 13:50
            '%Y %b %d %H:%M:%S',   # e.g 1997 Jan 1 13:50
            '%Y %B %d %H:%M:%S',   # e.g 1997 January 1 13:50
            '%Y/%b/%d %H:%M:%S',   # e.g 1997/Jan/1 13:50
            '%Y/%B/%d %H:%M:%S',   # e.g 1997/January/1 13:50
            '%Y-%b-%d %H:%M:%S',   # e.g 1997-Jan-1 13:50
            '%Y-%B-%d %H:%M:%S',   # e.g 1997-January-1 13:50
            '%b-%d-%Y %H:%M:%S',   # e.g Jan-1-1997 13:50
            '%B-%d-%Y %H:%M:%S',   # e.g January-1-1997 13:50
            '%b/%d/%Y %H:%M:%S',   # e.g Jan/1/1997 13:50
            '%B/%d/%Y %H:%M:%S',   # e.g January/1/1997 13:50
        ]

        for format in formats:
            for second in seconds:
                assert DateParser.parse(second.strftime(format)) == datetime.timestamp(second), second.strftime(format)

    def test_nyc_tz(self):

        # edt times
        assert 1666355040 == DateParser.parse('2022-10-21t08:24:00NYC')
        assert 1666355040 == DateParser.parse('2022-10-21t08:24:00EST')

        # est times
        assert 1671629040 == DateParser.parse('2022-12-21t08:24:00NYC')
        assert 1671629040 == DateParser.parse('2022-12-21t08:24:00EDT')

    def test_foo(self):
        # @DateParser.parse('january 1, 1997')
        DateParser.parse('2020-02t13:17:19Z')
        DateParser.parse('2020-02t13:17:19+0000')
        DateParser.parse('2020-02t13:17:19+00:00')
        DateParser.parse('2020-02t13:17:19+00')

    def test_is_month_valid(self):
        months = [
            'Jan', 'January',
            'Feb', 'February',
            'Mar', 'March',
            'Apr', 'April',
            'May',
            'Jun', 'June',
            'Jul', 'July',
            'Aug', 'August',
            'Sep', 'Sept', 'September',
            'Oct', 'October',
            'Nov', 'November',
            'Dec', 'December'
        ]

        for month in months:
            assert DateParser.is_month(month), month
            assert DateParser.is_month(month.lower()), month.lower()
            assert DateParser.is_month(month.upper()), month.upper()
            assert DateParser.is_month(' ' + month + ' '), month

    def test_is_month_invalid(self):
        months = [
            'not a month',
            'Jane',
            'Movember',
            '',
            '1',
        ]

        for month in months:
            assert not DateParser.is_month(month), month
            assert not DateParser.is_month(month.lower()), month.lower()
            assert not DateParser.is_month(month.upper()), month.upper()

    def test_is_year_valid(self):
        years = [
            '1000',
            '1999',
            '0001',
            '0000'
        ]

        for year in years:
            assert DateParser.is_year(year), year

    def test_is_year_invalid(self):
        years = [
            '000',
            'abcd',
            '10a0',
            '12345'
        ]

        for year in years:
            assert not DateParser.is_year(year), year
