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
from typed_python.lib.datetime.timezone import tz_abbr_to_utc_offset

ST_TIME = 1645812452  # feb 25, 2022 - Standard time
DS_TIME = 1661447252  # aug 25, 2022 - DST time


class TestTimezone(unittest.TestCase):

    def test_tz_abbr_ct(self):
        assert tz_abbr_to_utc_offset('ct', ST_TIME) == 21600
        assert tz_abbr_to_utc_offset('ct', DS_TIME) == 18000

    def test_tz_abbr_cdt(self):
        assert tz_abbr_to_utc_offset('cdt', ST_TIME) == 18000
        assert tz_abbr_to_utc_offset('cdt', DS_TIME) == 18000

    def test_tz_abbr_cst(self):
        assert tz_abbr_to_utc_offset('cst', ST_TIME) == 21600
        assert tz_abbr_to_utc_offset('cst', DS_TIME) == 21600

    def test_tz_abbr_est(self):
        assert tz_abbr_to_utc_offset('est', ST_TIME) == 18000
        assert tz_abbr_to_utc_offset('est', DS_TIME) == 18000

    def test_tz_abbr_edt(self):
        assert tz_abbr_to_utc_offset('edt', ST_TIME) == 14400
        assert tz_abbr_to_utc_offset('edt', DS_TIME) == 14400

    def test_tz_abbr_et(self):
        assert tz_abbr_to_utc_offset('et', ST_TIME) == 18000
        assert tz_abbr_to_utc_offset('et', DS_TIME) == 14400

    def test_tz_abbr_mt(self):
        assert tz_abbr_to_utc_offset('mt', ST_TIME) == 25200
        assert tz_abbr_to_utc_offset('mt', DS_TIME) == 21600

    def test_tz_abbr_mdt(self):
        assert tz_abbr_to_utc_offset('mdt', ST_TIME) == 21600
        assert tz_abbr_to_utc_offset('mdt', DS_TIME) == 21600

    def test_tz_abbr_mst(self):
        assert tz_abbr_to_utc_offset('mst', ST_TIME) == 25200
        assert tz_abbr_to_utc_offset('mst', DS_TIME) == 25200

    def test_tz_abbr_pt(self):
        assert tz_abbr_to_utc_offset('pt', ST_TIME) == 28800
        assert tz_abbr_to_utc_offset('pt', DS_TIME) == 25200

    def test_tz_abbr_pdt(self):
        assert tz_abbr_to_utc_offset('pdt', ST_TIME) == 25200
        assert tz_abbr_to_utc_offset('pdt', DS_TIME) == 25200

    def test_tz_abbr_pst(self):
        assert tz_abbr_to_utc_offset('pst', ST_TIME) == 28800
        assert tz_abbr_to_utc_offset('pst', DS_TIME) == 28800

    def test_tz_abbr_nyc(self):
        assert tz_abbr_to_utc_offset('nyc', ST_TIME) == 18000
        assert tz_abbr_to_utc_offset('nyc', DS_TIME) == 14400
        assert (tz_abbr_to_utc_offset('nyc', ST_TIME) ==
                tz_abbr_to_utc_offset('et', ST_TIME))
        assert (tz_abbr_to_utc_offset('nyc', DS_TIME) ==
                tz_abbr_to_utc_offset('et', DS_TIME))

    def test_tz_abbr_utc(self):
        assert tz_abbr_to_utc_offset('utc', ST_TIME) == 0
        assert tz_abbr_to_utc_offset('utc', DS_TIME) == 0

    def test_tz_abbr_gmt(self):
        assert tz_abbr_to_utc_offset('gmt', ST_TIME) == 0
        assert tz_abbr_to_utc_offset('gmt', DS_TIME) == 0

    def test_tz_abbr_z(self):
        assert tz_abbr_to_utc_offset('z', ST_TIME) == 0
        assert tz_abbr_to_utc_offset('z', DS_TIME) == 0
