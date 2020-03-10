#   Copyright 2017-2020 typed_python Authors
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

"""Code for sorting containers."""

from typed_python import ListOf, Entrypoint


def _sortAroundPivot(values, start, end, ixPivot, less):
    a = values[start]
    values[start] = values[ixPivot]
    values[ixPivot] = a

    pivot = values[start]
    i = start + 1
    j = start + 1

    while j <= end:
        if not less(pivot, values[j]):
            a = values[j]
            values[j] = values[i]
            values[i] = a
            i += 1
        j += 1

    a = values[start]
    values[start] = values[i - 1]
    values[i - 1] = a
    return i - 1


def _quicksortBetween(values, start, end, randomSeed, less):
    if end - start < 1:
        return

    newRandomSeed = 1013904223 + 1664525 * randomSeed

    ixPivot = start + (newRandomSeed) % (end - start + 1)

    i = _sortAroundPivot(values, start, end, ixPivot, less)
    i_top = i

    # if we have a block of equal values, make sure we don't accidentally have an
    # n^2 problem with it
    while i > start and values[i - 1] == values[i]:
        i -= 1

    while i_top < end and values[i_top] == values[i_top + 1]:
        i_top += 1

    _quicksortBetween(values, start, i - 1, newRandomSeed, less)
    _quicksortBetween(values, i_top + 1, end, newRandomSeed + 1, less)


@Entrypoint
def sort(values, key=None):
    """Perform an in-place sort on 'values', which must be a mutable sequence."""
    if len(values) <= 1:
        return

    if key is None:
        _quicksortBetween(values, 0, len(values) - 1, 1, lambda x, y: x < y)
    else:
        _quicksortBetween(values, 0, len(values) - 1, 1, lambda x, y: key(x) < key(y))


@Entrypoint
def sorted(values, key=None):
    valuesCopy = ListOf(type(values).ElementType)(values)
    sort(valuesCopy, key)
    return valuesCopy
