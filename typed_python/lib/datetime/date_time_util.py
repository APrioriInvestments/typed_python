from typed_python import Entrypoint, ListOf
from typed_python.lib.datetime.date_time import Date


@Entrypoint
def daterange(
    start: Date,
    end: Date,
    step: str = "days",
    stepSize: int = 1,
) -> ListOf(Date):
    """
    Build a list of dates between start and end.

    Parameters
    ----------
    start : Date
        Include this start date.
    end : Date
        Exclude this end date.
    step : OneOf("days", "weeks", "months", "years")
        The type of step to take.
    stepSize : int
        The number of increments in each step.
    """
    if step == "days":

        def next(dt: Date) -> Date:
            return dt.next(stepSize)

    elif step == "weeks":

        def next(dt: Date) -> Date:
            return dt.next(stepSize * 7)

    elif step == "months":

        maxDay = start.day

        def next(dt: Date) -> Date:
            return dt.nextMonth(stepSize, maxDay)

    elif step == "years":

        maxDay = start.day

        def next(dt: Date) -> Date:
            return dt.nextYear(stepSize, maxDay)

    else:
        raise Exception("step must be 'years', 'months', 'weeks', or 'days'")

    res = ListOf(Date)()
    while start < end:
        res.append(start)
        start = next(start)

    return res
