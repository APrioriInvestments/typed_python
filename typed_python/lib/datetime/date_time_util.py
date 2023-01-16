from typed_python import Entrypoint, ListOf
from typed_python.lib.datetime.date_time import Date


@Entrypoint
def daterange(
    start: Date, end: Date, step: str = "days", stepSize: int = 1, reversed=False
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

        if reversed:

            def iterate(dt: Date) -> Date:
                return dt.previous(stepSize)
        else:

            def iterate(dt: Date) -> Date:
                return dt.next(stepSize)

    elif step == "weeks":

        if reversed:

            def iterate(dt: Date) -> Date:
                return dt.previous(stepSize * 7)
        else:

            def iterate(dt: Date) -> Date:
                return dt.next(stepSize * 7)

    elif step == "months":

        maxDay = start.day

        if reversed:
            raise NotImplementedError("We haven't implemented walking backwards by months.")

        else:

            def iterate(dt: Date) -> Date:
                return dt.nextMonth(stepSize, maxDay)

    elif step == "years":

        maxDay = start.day

        if reversed:
            raise NotImplementedError("We haven't implemented walking backwards by years.")

        else:

            def iterate(dt: Date) -> Date:
                return dt.nextYear(stepSize, maxDay)

    else:
        raise Exception("step must be 'years', 'months', 'weeks', or 'days'")

    res = ListOf(Date)()
    if reversed:
        while start < end:
            res.append(end)
            end = iterate(end)

    else:
        while start < end:
            res.append(start)
            start = iterate(start)

    return res
