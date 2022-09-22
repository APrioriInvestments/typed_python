from typed_python import ListOf
from typed_python.lib.datetime.date_time_util import daterange
from typed_python.lib.datetime.date_time import Date


def test_daterange():
    # days by 1
    assert daterange(Date(2022, 1, 2), Date(2022, 1, 10)) == ListOf(Date)(
        [Date(2022, 1, day) for day in range(2, 10)]
    )
    # days by 2
    assert daterange(Date(2022, 1, 2), Date(2022, 1, 10), stepSize=2) == ListOf(Date)(
        [Date(2022, 1, day) for day in range(2, 10, 2)]
    )
    # months by 2
    ymd = [
        (2021, 12, 31),
        (2022, 2, 28),
        (2022, 4, 30),
        (2022, 6, 30),
        (2022, 8, 31),
        (2022, 10, 31),
        (2022, 12, 31),
    ]
    assert daterange(
        Date(2021, 12, 31), Date(2023, 1, 31), "months", stepSize=2
    ) == ListOf(Date)([Date(y, m, d) for (y, m, d) in ymd])
    # leap year
    ymd = [(2020, 2, 29), (2020, 5, 29)]
    assert daterange(
        Date(2020, 2, 29), Date(2020, 7, 31), "months", stepSize=3
    ) == ListOf(Date)([Date(y, m, d) for (y, m, d) in ymd])
    # leap year by year
    ymd = [(2020, 2, 29), (2022, 2, 28), (2024, 2, 29), (2026, 2, 28), (2028, 2, 29)]
    assert daterange(
        Date(2020, 2, 29), Date(2028, 7, 31), "years", stepSize=2
    ) == ListOf(Date)([Date(y, m, d) for (y, m, d) in ymd])

    # in reverse
    assert daterange(Date(2022, 1, 2), Date(2022, 1, 10), reversed=True) == ListOf(Date)(
        [Date(2022, 1, day) for day in range(10, 2, -1)]
    )
