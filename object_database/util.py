import logging

def formatTable(rows):
    rows = [[str(r) for r in row] for row in rows]

    cols = [[r[i] for r in rows] for i in range(len(rows[0]))]
    colWidth = [max([len(c) for c in col]) for col in cols]

    formattedRows = [
        "  ".join(row[col] + " " * (colWidth[col] - len(row[col])) for col in range(len(cols)))
            for row in rows
        ]
    formattedRows = formattedRows[:1] + [
        "  ".join("-" * colWidth[col] for col in range(len(cols)))
        ] + formattedRows[1:]

    return "\n".join(formattedRows)

def configureLogging(preamble=""):
    logging.basicConfig(format='[%(asctime)s] %(levelname)6s %(filename)30s:%(lineno)4s' 
        + ("|" + preamble if preamble else '') 
        + '| %(message)s', level=logging.ERROR
        )

def secondsToHumanReadable(seconds):
    if seconds < 120:
        return "%.2f seconds" % (seconds)
    if seconds < 120 * 60:
        return "%.2f minutes" % (seconds / 60)
    if seconds < 120 * 60 * 24:
        return "%.2f hours" % (seconds / 60 / 60)
    return "%.2f days" % (seconds / 60 / 60 / 24)

