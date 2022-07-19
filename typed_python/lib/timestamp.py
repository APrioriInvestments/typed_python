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

from typed_python.compiler.runtime import Entrypoint
from typed_python._types import civilfromdays, secondsfromcivil
from typed_python import Class, Final, Member, NamedTuple, Held
from time import strftime


import re
import string
import time
from calendar import monthrange
from io import StringIO

import six
from six import binary_type, text_type

from decimal import Decimal

Date = NamedTuple(year=int, month=int, day=int, hour=int, minute=int, second=int, ms=int, us=int, ns=int, weekday=int, doy=int)

DEFAULT_UTC_OFFSET = 0


"""
  Adapted from dateutils/parser
"""


class _timelex(object):
    # Fractional seconds are sometimes split by a comma
    _split_decimal = re.compile("([.,])")

    def __init__(self, instream):
        if six.PY2:
            # In Python 2, we can't duck type properly because unicode has
            # a 'decode' function, and we'd be double-decoding
            if isinstance(instream, (binary_type, bytearray)):
                instream = instream.decode()
        else:
            if getattr(instream, 'decode', None) is not None:
                instream = instream.decode()

        if isinstance(instream, text_type):
            instream = StringIO(instream)
        elif getattr(instream, 'read', None) is None:
            raise TypeError('Parser must be a string or character stream, not '
                            '{itype}'.format(itype=instream.__class__.__name__))

        self.instream = instream
        self.charstack = []
        self.tokenstack = []
        self.eof = False

    def get_token(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.

        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """
        if self.tokenstack:
            return self.tokenstack.pop(0)

        seenletters = False
        token = None
        state = None

        while not self.eof:
            # We only realize that we've reached the end of a token when we
            # find a character that's not part of the current token - since
            # that character may be part of the next token, it's stored in the
            # charstack.
            if self.charstack:
                nextchar = self.charstack.pop(0)
            else:
                nextchar = self.instream.read(1)
                while nextchar == '\x00':
                    nextchar = self.instream.read(1)

            if not nextchar:
                self.eof = True
                break
            elif not state:
                # First character of the token - determines if we're starting
                # to parse a word, a number or something else.
                token = nextchar
                if self.isword(nextchar):
                    state = 'a'
                elif self.isnum(nextchar):
                    state = '0'
                elif self.isspace(nextchar):
                    token = ' '
                    break  # emit token
                else:
                    break  # emit token
            elif state == 'a':
                # If we've already started reading a word, we keep reading
                # letters until we find something that's not part of a word.
                seenletters = True
                if self.isword(nextchar):
                    token += nextchar
                elif nextchar == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == '0':
                # If we've already started reading a number, we keep reading
                # numbers until we find something that doesn't fit.
                if self.isnum(nextchar):
                    token += nextchar
                elif nextchar == '.' or (nextchar == ',' and len(token) >= 2):
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == 'a.':
                # If we've seen some letters and a dot separator, continue
                # parsing, and the tokens will be broken up later.
                seenletters = True
                if nextchar == '.' or self.isword(nextchar):
                    token += nextchar
                elif self.isnum(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == '0.':
                # If we've seen at least one dot separator, keep going, we'll
                # break up the tokens later.
                if nextchar == '.' or self.isnum(nextchar):
                    token += nextchar
                elif self.isword(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token

        if (state in ('a.', '0.') and (seenletters or token.count('.') > 1 or
                                       token[-1] in '.,')):
            l = self._split_decimal.split(token)
            token = l[0]
            for tok in l[1:]:
                if tok:
                    self.tokenstack.append(tok)

        if state == '0.' and token.count('.') == 0:
            token = token.replace(',', '.')

        return token

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token is None:
            raise StopIteration

        return token

    def next(self):
        return self.__next__()  # Python 2.x support

    @classmethod
    def split(cls, s):
        return list(cls(s))

    @classmethod
    def isword(cls, nextchar):
        """ Whether or not the next character is part of a word """
        return nextchar.isalpha()

    @classmethod
    def isnum(cls, nextchar):
        """ Whether the next character is part of a number """
        return nextchar.isdigit()

    @classmethod
    def isspace(cls, nextchar):
        """ Whether the next character is whitespace """
        return nextchar.isspace()


class _resultbase(object):

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def _repr(self, classname):
        l = []
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                l.append("%s=%s" % (attr, repr(value)))
        return "%s(%s)" % (classname, ", ".join(l))

    def __len__(self):
        return (sum(getattr(self, attr) is not None
                    for attr in self.__slots__))

    def __repr__(self):
        return self._repr(self.__class__.__name__)


class _parserinfo(object):
    """
    Class which handles what inputs are accepted. Subclass this to customize
    the language and acceptable values for each parameter.

    :param dayfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the day (``True``) or month (``False``). If
        ``yearfirst`` is set to ``True``, this distinguishes between YDM
        and YMD. Default is ``False``.

    :param yearfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the year. If ``True``, the first number is taken
        to be the year, otherwise the last number is taken to be the year.
        Default is ``False``.
    """

    # m from a.m/p.m, t from ISO T separator
    JUMP = [" ", ".", ",", ";", "-", "/", "'",
            "at", "on", "and", "ad", "m", "t", "of",
            "st", "nd", "rd", "th"]

    WEEKDAYS = [("Mon", "Monday"),
                ("Tue", "Tuesday"),     # TODO: "Tues"
                ("Wed", "Wednesday"),
                ("Thu", "Thursday"),    # TODO: "Thurs"
                ("Fri", "Friday"),
                ("Sat", "Saturday"),
                ("Sun", "Sunday")]
    MONTHS = [("Jan", "January"),
              ("Feb", "February"),      # TODO: "Febr"
              ("Mar", "March"),
              ("Apr", "April"),
              ("May", "May"),
              ("Jun", "June"),
              ("Jul", "July"),
              ("Aug", "August"),
              ("Sep", "Sept", "September"),
              ("Oct", "October"),
              ("Nov", "November"),
              ("Dec", "December")]
    HMS = [("h", "hour", "hours"),
           ("m", "minute", "minutes"),
           ("s", "second", "seconds")]
    AMPM = [("am", "a"),
            ("pm", "p")]
    UTCZONE = ["UTC", "GMT", "Z"]
    PERTAIN = ["of"]
    TZOFFSET = {}
    # TODO: ERA = ["AD", "BC", "CE", "BCE", "Stardate",
    #              "Anno Domini", "Year of Our Lord"]

    def __init__(self, dayfirst=False, yearfirst=False):
        self._jump = self._convert(self.JUMP)
        self._weekdays = self._convert(self.WEEKDAYS)
        self._months = self._convert(self.MONTHS)
        self._hms = self._convert(self.HMS)
        self._ampm = self._convert(self.AMPM)
        self._utczone = self._convert(self.UTCZONE)
        self._pertain = self._convert(self.PERTAIN)

        self.dayfirst = dayfirst
        self.yearfirst = yearfirst

        self._year = time.localtime().tm_year
        self._century = self._year // 100 * 100

    def _convert(self, lst):
        dct = {}
        for i, v in enumerate(lst):
            if isinstance(v, tuple):
                for v in v:
                    dct[v.lower()] = i
            else:
                dct[v.lower()] = i
        return dct

    def jump(self, name):
        return name.lower() in self._jump

    def weekday(self, name):
        try:
            return self._weekdays[name.lower()]
        except KeyError:
            pass
        return None

    def month(self, name):
        try:
            return self._months[name.lower()] + 1
        except KeyError:
            pass
        return None

    def hms(self, name):
        try:
            return self._hms[name.lower()]
        except KeyError:
            return None

    def ampm(self, name):
        try:
            return self._ampm[name.lower()]
        except KeyError:
            return None

    def pertain(self, name):
        return name.lower() in self._pertain

    def utczone(self, name):
        return name.lower() in self._utczone

    def tzoffset(self, name):
        if name in self._utczone:
            return 0

        return self.TZOFFSET.get(name)

    def convertyear(self, year, century_specified=False):
        """
        Converts two-digit years to year within [-50, 49]
        range of self._year (current local time)
        """

        # Function contract is that the year is always positive
        assert year >= 0

        if year < 100 and not century_specified:
            # assume current century to start
            year += self._century

            if year >= self._year + 50:  # if too far in future
                year -= 100
            elif year < self._year - 50:  # if too far in past
                year += 100

        return year

    def validate(self, res):
        # move to info
        if res.year is not None:
            res.year = self.convertyear(res.year, res.century_specified)

        if res.tzoffset == 0 and not res.tzname or res.tzname == 'Z':
            res.tzname = "UTC"
            res.tzoffset = 0
        elif res.tzoffset != 0 and res.tzname and self.utczone(res.tzname):
            res.tzoffset = 0
        return True


class _ymd(list):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.century_specified = False
        self.dstridx = None
        self.mstridx = None
        self.ystridx = None

    @property
    def has_year(self):
        return self.ystridx is not None

    @property
    def has_month(self):
        return self.mstridx is not None

    @property
    def has_day(self):
        return self.dstridx is not None

    def could_be_day(self, value):
        if self.has_day:
            return False
        elif not self.has_month:
            return 1 <= value <= 31
        elif not self.has_year:
            # Be permissive, assume leapyear
            month = self[self.mstridx]
            return 1 <= value <= monthrange(2000, month)[1]
        else:
            month = self[self.mstridx]
            year = self[self.ystridx]
            return 1 <= value <= monthrange(year, month)[1]

    def append(self, val, label=None):
        if hasattr(val, '__len__'):
            if val.isdigit() and len(val) > 2:
                self.century_specified = True
                if label not in [None, 'Y']:  # pragma: no cover
                    raise ValueError(label)
                label = 'Y'
        elif val > 100:
            self.century_specified = True
            if label not in [None, 'Y']:  # pragma: no cover
                raise ValueError(label)
            label = 'Y'

        super(self.__class__, self).append(int(val))

        if label == 'M':
            if self.has_month:
                raise ValueError('Month is already set')
            self.mstridx = len(self) - 1
        elif label == 'D':
            if self.has_day:
                raise ValueError('Day is already set')
            self.dstridx = len(self) - 1
        elif label == 'Y':
            if self.has_year:
                raise ValueError('Year is already set')
            self.ystridx = len(self) - 1

    def _resolve_from_stridxs(self, strids):
        """
        Try to resolve the identities of year/month/day elements using
        ystridx, mstridx, and dstridx, if enough of these are specified.
        """
        if len(self) == 3 and len(strids) == 2:
            # we can back out the remaining stridx value
            missing = [x for x in range(3) if x not in strids.values()]
            key = [x for x in ['y', 'm', 'd'] if x not in strids]
            assert len(missing) == len(key) == 1
            key = key[0]
            val = missing[0]
            strids[key] = val

        assert len(self) == len(strids)  # otherwise this should not be called
        out = {key: self[strids[key]] for key in strids}
        return (out.get('y'), out.get('m'), out.get('d'))

    def resolve_ymd(self, yearfirst, dayfirst):
        len_ymd = len(self)
        year, month, day = (None, None, None)

        strids = (('y', self.ystridx),
                  ('m', self.mstridx),
                  ('d', self.dstridx))

        strids = {key: val for key, val in strids if val is not None}
        if (len(self) == len(strids) > 0 or
                (len(self) == 3 and len(strids) == 2)):
            return self._resolve_from_stridxs(strids)

        mstridx = self.mstridx

        if len_ymd > 3:
            raise ValueError("More than three YMD values")
        elif len_ymd == 1 or (mstridx is not None and len_ymd == 2):
            # One member, or two members with a month string
            if mstridx is not None:
                month = self[mstridx]
                # since mstridx is 0 or 1, self[mstridx-1] always
                # looks up the other element
                other = self[mstridx - 1]
            else:
                other = self[0]

            if len_ymd > 1 or mstridx is None:
                if other > 31:
                    year = other
                else:
                    day = other

        elif len_ymd == 2:
            # Two members with numbers
            if self[0] > 31:
                # 99-01
                year, month = self
            elif self[1] > 31:
                # 01-99
                month, year = self
            elif dayfirst and self[1] <= 12:
                # 13-01
                day, month = self
            else:
                # 01-13
                month, day = self

        elif len_ymd == 3:
            # Three members
            if mstridx == 0:
                if self[1] > 31:
                    # Apr-2003-25
                    month, year, day = self
                else:
                    month, day, year = self
            elif mstridx == 1:
                if self[0] > 31 or (yearfirst and self[2] <= 31):
                    # 99-Jan-01
                    year, month, day = self
                else:
                    # 01-Jan-01
                    # Give precendence to day-first, since
                    # two-digit years is usually hand-written.
                    day, month, year = self

            elif mstridx == 2:
                # WTF!?
                if self[1] > 31:
                    # 01-99-Jan
                    day, year, month = self
                else:
                    # 99-01-Jan
                    year, day, month = self

            else:
                if (self[0] > 31 or
                    self.ystridx == 0 or
                        (yearfirst and self[1] <= 12 and self[2] <= 31)):
                    # 99-01-01
                    if dayfirst and self[2] <= 12:
                        year, day, month = self
                    else:
                        year, month, day = self
                elif self[0] > 12 or (dayfirst and self[1] <= 12):
                    # 13-01-01
                    day, month, year = self
                else:
                    # 01-13-01
                    month, day, year = self

        return year, month, day


class _parser:
    class _result(_resultbase):
        __slots__ = ["year", "month", "day", "weekday",
                     "hour", "minute", "second", "microsecond",
                     "tzname", "tzoffset", "ampm", "any_unused_tokens"]

    def parse(self, timestr):
        """
        :param timestr:
            The string to parse.
        """
        info = parserinfo_instance
        dayfirst = info.dayfirst
        yearfirst = info.yearfirst

        res = self._result()
        l = _timelex.split(timestr)         # Splits the timestr into tokens

        skipped_idxs = []

        # year/month/day list
        ymd = _ymd()

        len_l = len(l)
        i = 0

        try:
            while i < len_l:
                # Check if it's a number
                value_repr = l[i]
                try:
                    value = float(value_repr)
                except ValueError:
                    value = None

                if value is not None:
                    # Numeric token
                    i = self._parse_numeric_token(l, i, info, ymd, res)

                # Check weekday
                elif info.weekday(l[i]) is not None:
                    value = info.weekday(l[i])
                    res.weekday = value

                # Check month name
                elif info.month(l[i]) is not None:
                    value = info.month(l[i])
                    ymd.append(value, 'M')

                    if i + 1 < len_l:
                        if l[i + 1] in ('-', '/'):
                            # Jan-01[-99]
                            sep = l[i + 1]
                            ymd.append(l[i + 2])

                            if i + 3 < len_l and l[i + 3] == sep:
                                # Jan-01-99
                                ymd.append(l[i + 4])
                                i += 2

                            i += 2

                        elif (i + 4 < len_l and l[i + 1] == l[i + 3] == ' ' and
                              info.pertain(l[i + 2])):
                            # Jan of 01
                            # In this case, 01 is clearly year
                            if l[i + 4].isdigit():
                                # Convert it here to become unambiguous
                                value = int(l[i + 4])
                                year = str(info.convertyear(value))
                                ymd.append(year, 'Y')
                            else:
                                # Wrong guess
                                pass
                                # TODO: not hit in tests
                            i += 4

                # Check am/pm
                elif info.ampm(l[i]) is not None:
                    value = info.ampm(l[i])
                    val_is_ampm = self._ampm_valid(res.hour, res.ampm)

                    if val_is_ampm:
                        res.hour = self._adjust_ampm(res.hour, value)
                        res.ampm = value

                # Check for a timezone name
                elif self._could_be_tzname(res.hour, res.tzname, res.tzoffset, l[i]):
                    res.tzname = l[i]
                    res.tzoffset = info.tzoffset(res.tzname)

                    # Check for something like GMT+3, or BRST+3. Notice
                    # that it doesn't mean "I am 3 hours after GMT", but
                    # "my time +3 is GMT". If found, we reverse the
                    # logic so that timezone parsing code will get it
                    # right.
                    if i + 1 < len_l and l[i + 1] in ('+', '-'):
                        l[i + 1] = ('+', '-')[l[i + 1] == '+']
                        res.tzoffset = None
                        if info.utczone(res.tzname):
                            # With something like GMT+3, the timezone
                            # is *not* GMT.
                            res.tzname = None

                # Check for a numbered timezone
                elif res.hour is not None and l[i] in ('+', '-'):
                    signal = (-1, 1)[l[i] == '+']
                    len_li = len(l[i + 1])

                    # TODO: check that l[i + 1] is integer?
                    if len_li == 4:
                        # -0300
                        hour_offset = int(l[i + 1][:2])
                        min_offset = int(l[i + 1][2:])
                    elif i + 2 < len_l and l[i + 2] == ':':
                        # -03:00
                        hour_offset = int(l[i + 1])
                        min_offset = int(l[i + 3])  # TODO: Check that l[i+3] is minute-like?
                        i += 2
                    elif len_li <= 2:
                        # -[0]3
                        hour_offset = int(l[i + 1][:2])
                        min_offset = 0
                    else:
                        raise ValueError(timestr)

                    res.tzoffset = signal * (hour_offset * 3600 + min_offset * 60)

                    # Look for a timezone name between parenthesis
                    if (i + 5 < len_l and
                            info.jump(l[i + 2]) and l[i + 3] == '(' and
                            l[i + 5] == ')' and
                            3 <= len(l[i + 4]) and
                            self._could_be_tzname(res.hour, res.tzname,
                                                  None, l[i + 4])):
                        # -0300 (BRST)
                        res.tzname = l[i + 4]
                        i += 4

                    i += 1

                # Check jumps
                elif not (info.jump(l[i])):
                    raise ValueError(timestr)

                else:
                    skipped_idxs.append(i)
                i += 1

            # Process year/month/day
            year, month, day = ymd.resolve_ymd(yearfirst, dayfirst)

            res.century_specified = ymd.century_specified
            res.year = year
            res.month = month
            res.day = day

        except (IndexError, ValueError):
            return None

        if not info.validate(res):
            return None

        return Timestamp.from_date(
            res.year,
            res.month,
            res.day,
            res.hour,
            res.minute,
            res.second,
            res.microsecond,
            0
        )

    def _parse_numeric_token(self, tokens, idx, info, ymd, res):
        # Token is a number
        value_repr = tokens[idx]
        try:
            value = self._to_decimal(value_repr)
        except Exception as e:
            six.raise_from(ValueError('Unknown numeric token'), e)

        len_li = len(value_repr)

        len_l = len(tokens)

        if (len(ymd) == 3 and len_li in (2, 4) and
            res.hour is None and
            (idx + 1 >= len_l or
             (tokens[idx + 1] != ':' and
              info.hms(tokens[idx + 1]) is None))):
            # 19990101T23[59]
            s = tokens[idx]
            res.hour = int(s[:2])

            if len_li == 4:
                res.minute = int(s[2:])

        elif len_li == 6 or (len_li > 6 and tokens[idx].find('.') == 6):
            # YYMMDD or HHMMSS[.ss]
            s = tokens[idx]

            if not ymd and '.' not in tokens[idx]:
                ymd.append(s[:2])
                ymd.append(s[2:4])
                ymd.append(s[4:])
            else:
                # 19990101T235959[.59]

                # TODO: Check if res attributes already set.
                res.hour = int(s[:2])
                res.minute = int(s[2:4])
                res.second, res.microsecond = self._parsems(s[4:])

        elif len_li in (8, 12, 14):
            # YYYYMMDD
            s = tokens[idx]
            ymd.append(s[:4], 'Y')
            ymd.append(s[4:6])
            ymd.append(s[6:8])

            if len_li > 8:
                res.hour = int(s[8:10])
                res.minute = int(s[10:12])

                if len_li > 12:
                    res.second = int(s[12:])

        elif self._find_hms_idx(idx, tokens, info, allow_jump=True) is not None:
            # HH[ ]h or MM[ ]m or SS[.ss][ ]s
            hms_idx = self._find_hms_idx(idx, tokens, info, allow_jump=True)
            (idx, hms) = self._parse_hms(idx, tokens, info, hms_idx)
            if hms is not None:
                # TODO: checking that hour/minute/second are not
                # already set?
                self._assign_hms(res, value_repr, hms)

        elif idx + 2 < len_l and tokens[idx + 1] == ':':
            # HH:MM[:SS[.ss]]
            res.hour = int(value)
            value = self._to_decimal(tokens[idx + 2])  # TODO: try/except for this?
            (res.minute, res.second) = self._parse_min_sec(value)

            if idx + 4 < len_l and tokens[idx + 3] == ':':
                res.second, res.microsecond = self._parsems(tokens[idx + 4])

                idx += 2

            idx += 2

        elif idx + 1 < len_l and tokens[idx + 1] in ('-', '/', '.'):
            sep = tokens[idx + 1]
            ymd.append(value_repr)

            if idx + 2 < len_l and not info.jump(tokens[idx + 2]):
                if tokens[idx + 2].isdigit():
                    # 01-01[-01]
                    ymd.append(tokens[idx + 2])
                else:
                    # 01-Jan[-01]
                    value = info.month(tokens[idx + 2])

                    if value is not None:
                        ymd.append(value, 'M')
                    else:
                        raise ValueError()

                if idx + 3 < len_l and tokens[idx + 3] == sep:
                    # We have three members
                    value = info.month(tokens[idx + 4])

                    if value is not None:
                        ymd.append(value, 'M')
                    else:
                        ymd.append(tokens[idx + 4])
                    idx += 2

                idx += 1
            idx += 1

        elif idx + 1 >= len_l or info.jump(tokens[idx + 1]):
            if idx + 2 < len_l and info.ampm(tokens[idx + 2]) is not None:
                # 12 am
                hour = int(value)
                res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 2]))
                idx += 1
            else:
                # Year, month or day
                ymd.append(value)
            idx += 1

        elif info.ampm(tokens[idx + 1]) is not None and (0 <= value < 24):
            # 12am
            hour = int(value)
            res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 1]))
            idx += 1

        elif ymd.could_be_day(value):
            ymd.append(value)

        return idx

    def _find_hms_idx(self, idx, tokens, info, allow_jump):
        len_l = len(tokens)

        if idx + 1 < len_l and info.hms(tokens[idx + 1]) is not None:
            # There is an "h", "m", or "s" label following this token.  We take
            # assign the upcoming label to the current token.
            # e.g. the "12" in 12h"
            hms_idx = idx + 1

        elif (allow_jump and idx + 2 < len_l and tokens[idx + 1] == ' ' and
              info.hms(tokens[idx + 2]) is not None):
            # There is a space and then an "h", "m", or "s" label.
            # e.g. the "12" in "12 h"
            hms_idx = idx + 2

        elif idx > 0 and info.hms(tokens[idx - 1]) is not None:
            # There is a "h", "m", or "s" preceeding this token.  Since neither
            # of the previous cases was hit, there is no label following this
            # token, so we use the previous label.
            # e.g. the "04" in "12h04"
            hms_idx = idx - 1

        elif (1 < idx == len_l - 1 and tokens[idx - 1] == ' ' and
              info.hms(tokens[idx - 2]) is not None):
            # If we are looking at the final token, we allow for a
            # backward-looking check to skip over a space.
            # TODO: Are we sure this is the right condition here?
            hms_idx = idx - 2

        else:
            hms_idx = None

        return hms_idx

    def _assign_hms(self, res, value_repr, hms):
        # See GH issue #427, fixing float rounding
        value = self._to_decimal(value_repr)

        if hms == 0:
            # Hour
            res.hour = int(value)
            if value % 1:
                res.minute = int(60 * (value % 1))

        elif hms == 1:
            (res.minute, res.second) = self._parse_min_sec(value)

        elif hms == 2:
            (res.second, res.microsecond) = self._parsems(value_repr)

    def _could_be_tzname(self, hour, tzname, tzoffset, token):
        return (hour is not None and
                tzname is None and
                tzoffset is None and
                len(token) <= 5 and
                all(x in string.ascii_uppercase for x in token))

    def _ampm_valid(self, hour, ampm):
        """
        For fuzzy parsing, 'a' or 'am' (both valid English words)
        may erroneously trigger the AM/PM flag. Deal with that
        here.
        """
        val_is_ampm = True

        # If AM/PM is found and hour is not, raise a ValueError
        if hour is None:
            raise ValueError('No hour specified with AM or PM flag.')
        elif not 0 <= hour <= 12:
            raise ValueError('Invalid hour specified for 12-hour clock.')

        return val_is_ampm

    def _adjust_ampm(self, hour, ampm):
        if hour < 12 and ampm == 1:
            hour += 12
        elif hour == 12 and ampm == 0:
            hour = 0
        return hour

    def _parse_min_sec(self, value):
        # TODO: Every usage of this function sets res.second to the return
        # value. Are there any cases where second will be returned as None and
        # we *dont* want to set res.second = None?
        minute = int(value)
        second = None

        sec_remainder = value % 1
        if sec_remainder:
            second = int(60 * sec_remainder)
        return (minute, second)

    def _parsems(self, value):
        """Parse a I[.F] seconds value into (seconds, microseconds)."""
        if "." not in value:
            return int(value), 0
        else:
            i, f = value.split(".")
            return int(i), int(f.ljust(6, "0")[:6])

    def _parse_hms(self, idx, tokens, info, hms_idx):
        # TODO: Is this going to admit a lot of false-positives for when we
        # just happen to have digits and "h", "m" or "s" characters in non-date
        # text?  I guess hex hashes won't have that problem, but there's plenty
        # of random junk out there.
        if hms_idx is None:
            hms = None
            new_idx = idx
        elif hms_idx > idx:
            hms = info.hms(tokens[hms_idx])
            new_idx = hms_idx
        else:
            # Looking backwards, increment one.
            hms = info.hms(tokens[hms_idx]) + 1
            new_idx = idx

        return (new_idx, hms)

    def _to_decimal(self, val):
        try:
            decimal_value = Decimal(val)
            # See GH 662, edge case, infinite value should not be converted via `_to_decimal`
            if not decimal_value.is_finite():
                raise ValueError("Converted decimal value is infinite or NaN")
        except Exception as e:
            msg = "Could not convert %s to decimal" % val
            six.raise_from(ValueError(msg), e)
        else:
            return decimal_value


class UnknownTimezoneWarning(RuntimeWarning):
    """Raised when the parser finds a timezone it cannot parse into a tzinfo"""


def string_offset_to_seconds(utc_offset: str) -> int:
    offset = ''.join(utc_offset.split(':'))
    hrs = int(offset[0:3])
    mins = int(offset[3:5])
    return hrs * 3600 + (mins * 60 if hrs > 0 else mins * -60)


parserinfo_instance = _parserinfo()
parser_instance = _parser()


@Held
class Timestamp(Class, Final):
    ts = Member(float)

    @staticmethod
    def make(ts):

        return Timestamp(ts=ts)

    def __int__(self):
        return int(self.ts)

    def __float__(self):
        return self.ts

    def __str__(self):
        return self.to_string()

    @Entrypoint
    def date(self, utc_offset: int = DEFAULT_UTC_OFFSET) -> Date:
        '''
        Creates a Date tuple from this timestamp representing a date
        Parameters:
            utc_offset (int): The offset fromt UTC in seconds

        Returns:
            date (Date): a Date tuple. E.g. Date(year=2022, month=6, day=1, hour=5, minute=45, second=30, ms=1, us=4, ns=33)
        '''
        utc_offset = DEFAULT_UTC_OFFSET if utc_offset is None else utc_offset
        ts = self.ts + utc_offset

        cfd = civilfromdays(ts)

        return Date(
            year=cfd[0],
            month=cfd[1],
            day=cfd[2],
            hour=cfd[3],
            minute=cfd[4],
            second=cfd[5],
            ms=cfd[6],
            us=cfd[7],
            ns=cfd[8],
            weekday=cfd[9],
            doy=cfd[10]
        )

    @Entrypoint  # noqa : F811
    def date(self, utc_offset: str = DEFAULT_UTC_OFFSET) -> Date:
        '''
        Creates a Date tuple from this timestamp representing a date
        Parameters:
            utc_offset (string): The offset from UTC as a string. E.g. '+0200' or '+02:00'

        Returns:
            date (Date): a Date tuple. E.g. Date(year=2022, month=6, day=1, hour=5, minute=45, second=30, ms=1, us=4, ns=33)
        '''
        return self.date(string_offset_to_seconds(utc_offset))

    @Entrypoint
    def to_string(self, utc_offset: int = DEFAULT_UTC_OFFSET, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        '''
        Converts a Timestamp to a string in a given format
        Parameters:
            utc_offset (int): The offset from UTC in seconds
            format (str): A string specifying formatting directives. E.g. '%Y-%m-%d %H:%M:%S'
        Returns:
            date_str(str): A string represening the date in the specified format. E.g. "Mon January 2, 2021"
        '''
        d = self.date(utc_offset)
        return strftime(format, (d.year, d.month, d.day, d.hour, d.minute, d.second, d.weekday, d.doy, 0))

    @Entrypoint  # noqa : F811
    def to_string(self, utc_offset: str = DEFAULT_UTC_OFFSET, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        '''
        Converts a Timestamp to a string in a given format
        Parameters:
            utc_offset (string): The offset from UTC as a string. E.g. '+0200' or '+02:00'
            format (str): A string specifying formatting directives. E.g. '%Y-%m-%d %H:%M:%S'
        Returns:
            date_str(str): A string representing the date in the specified format. E.g. "Mon January 2, 2021"
        '''
        date = self.date(utc_offset, format)
        return f"{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:{date.minute:02d}:{date.second:02d}"

    @Entrypoint
    @staticmethod
    def from_date(year=0, month=0, day=0, hour=0, minute=0, second=0, ms=0, us=0, ns=0):
        '''
        Creates a Timestamp from date values.
        Parameters:
            year (int): The year
            month (int): The month. January: 1, February: 2, ....
            day (int): The day
            hour (int): The hour (0-23)
            minute (int): The minute
            second (int): The second
            ms (int): The millisecond
            us (int): The microsecond
            ns(int): The nanosecond
        Returns:
            timestamp (Timestamp): A Timestamp
        '''

        ts = secondsfromcivil(
            year,
            month,
            day,
            hour,
            minute,
            second,
            ms,
            us,
            ns
        )
        return Timestamp(ts=ts)

    @Entrypoint  # noqa : F811
    @staticmethod
    def parse(date_str: str, format: str = "%Y-%m-%d %H:%M:%S"):
        '''
        Creates a Timestamp from date values.
        Parameters:
            date_str (str): A date string
            format (str): a format string. e.g. "%Y-%m-%d %H:%M:%S". defaults to "%Y-%m-%d %H:%M:%S"
        Returns:
            timestamp (Timestamp): A Timestamp
        '''
        return parser_instance.parse(date_str)
