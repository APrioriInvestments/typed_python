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

import sys
from typed_python import sha_hash
from typed_python.compiler.global_variable_definition import GlobalVariableMetadata
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.typed_list_masquerading_as_list_wrapper import TypedListMasqueradingAsList

from typed_python import UInt8, Int32, ListOf, Tuple, TupleOf, Dict, Set, ConstDict
from typed_python import Class, Final, Member, pointerTo, PointerTo
from typed_python.type_promotion import isInteger

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

from typed_python.compiler.native_ast import VoidPtr


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def convertIterableToBytes(outPtr, iterable, canThrow):
    lst = ListOf(UInt8)()
    lst.reserve(len(iterable))

    try:
        for i in iterable:
            lst.append(i)

        # bytes from ListOf(UInt8) is guaranteed to work.
        outPtr.initialize(bytes(lst))

        return True
    except:  # noqa
        if canThrow:
            raise

        return False


def bytesJoinIterable(sep, iterable):
    """Converts the iterable container to list of bytes objects and calls sep.join(iterable).

    Compiling bytes.join on an iterable compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        sep: A bytes object to separate the items.
        iterable: Iterable container with bytes objects only.

    Returns:
        A bytes object with joined values.

    Raises:
        TypeError: If any of the values in the container is not of bytes type.
    """
    items = ListOf(bytes)()

    for item in iterable:
        if isinstance(item, bytes):
            items.append(item)
        else:
            raise TypeError("expected str instance")
    return sep.join(items)


IS_38_OR_LOWER = sys.version_info.minor <= 8


def bytes_replace(x: bytes, old: bytes, new: bytes, maxCount: int) -> bytes:
    """Given a bytes object, replaces old subsequence with new subsequence at most maxCount times.

    Compiling bytes.replace compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: A bytes object to operate on.
        old: The subsequence of bytes to search for.
        new: The replacement value for each found subsequence.
        maxCount: Replaces at most maxCount occurrences. Use -1 to indicate no limit.

    Returns:
        Adjusted bytes object with replacement(s).
    """
    if IS_38_OR_LOWER:
        # versions 3.8 and lower have a bug where b''.replace(b'', b'SOMETHING', 1) returns
        # the empty string.
        if maxCount == 0 or (maxCount >= 0 and len(x) == 0 and len(old) == 0):
            return x
    else:
        if maxCount == 0:
            return x

        if maxCount >= 0 and len(x) == 0 and len(old) == 0:
            return new

    accumulator = ListOf(bytes)()

    pos = 0
    seen = 0
    inc = 0 if len(old) else 1
    if len(old) == 0:
        accumulator.append(b'')
        seen += 1

    while True:
        if maxCount >= 0 and seen >= maxCount:
            nextLoc = -1
        else:
            nextLoc = x.find(old, pos)

        if nextLoc >= 0 and nextLoc < len(x):
            accumulator.append(x[pos:nextLoc + inc])

            if len(old):
                pos = nextLoc + len(old)
            else:
                pos += 1

            seen += 1
        else:
            accumulator.append(x[pos:])
            return new.join(accumulator)


def bytes_isalnum(x: bytes) -> bool:
    """Checks if given bytes object contains only alphanumeric elements.

    Compiling bytes.isalum compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.

    Returns:
        Result of check.
    """
    if len(x) == 0:
        return False
    for i in x:
        if not (ord('0') <= i <= ord('9') or ord('A') <= i <= ord('Z') or ord('a') <= i <= ord('z')):
            return False
    return True


def bytes_isalpha(x: bytes) -> bool:
    """Checks if given bytes object contains only alphabetic elements.

    Compiling bytes.isalpha compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.

    Returns:
        Result of check.
    """
    if len(x) == 0:
        return False
    for i in x:
        if not (ord('A') <= i <= ord('Z') or ord('a') <= i <= ord('z')):
            return False
    return True


def bytes_isdigit(x: bytes) -> bool:
    """Checks if given bytes object contains only digit elements.

    Compiling bytes.isdigit compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.

    Returns:
        Result of check.
    """
    if len(x) == 0:
        return False
    for i in x:
        if not (ord('0') <= i <= ord('9')):
            return False
    return True


def bytes_islower(x: bytes) -> bool:
    """Checks if given bytes object contains only lowercase elements.

    Compiling bytes.islower compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.

    Returns:
        Result of check.
    """
    found_lower = False
    for i in x:
        if ord('a') <= i <= ord('z'):
            found_lower = True
        elif ord('A') <= i <= ord('Z'):
            return False
    return found_lower


def bytes_isspace(x: bytes) -> bool:
    """Checks if given bytes object contains only whitespace elements.

    Compiling bytes.isspace compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.

    Returns:
        Result of check.
    """
    if len(x) == 0:
        return False
    for i in x:
        if i != ord(' ') and i != ord('\t') and i != ord('\n') and i != ord('\r') and i != 0x0b and i != ord('\f'):
            return False
    return True


def bytes_istitle(x: bytes) -> bool:
    """Checks if given bytes object contains only titlecase elements.

    Compiling bytes.istitle compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.

    Returns:
        Result of check.
    """
    if len(x) == 0:
        return False
    last_cased = False
    found_one = False
    for i in x:
        upper = ord('A') <= i <= ord('Z')
        lower = ord('a') <= i <= ord('z')
        if upper and last_cased:
            return False
        if lower and not last_cased:
            return False
        last_cased = upper or lower
        if last_cased:
            found_one = True
    return found_one


def bytes_isupper(x: bytes) -> bool:
    """Checks if given bytes object contains only uppercase elements.

    Compiling bytes.isupper compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.

    Returns:
        Result of check.
    """
    found_upper = False
    for i in x:
        if ord('A') <= i <= ord('Z'):
            found_upper = True
        elif ord('a') <= i <= ord('z'):
            return False
    return found_upper


def bytes_startswith(x: bytes, prefix: bytes) -> bool:
    """Does given bytes object start with the subsequence prefix?

    Compiling bytes.startswith, with no range arguments, compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.
        prefix: The subsequence to look for.

    Returns:
        Result of check.
    """
    if len(x) < len(prefix):
        return False
    index = 0
    for i in prefix:
        if x[index] != i:
            return False
        index += 1
    return True


def bytes_startswith_range(x: bytes, prefix: bytes, start: int, end: int) -> bool:
    """Does specified slice of given bytes object start with the subsequence prefix?

    Compiling bytes.startswith with range arguments compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.
        prefix: The subsequence to look for.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Result of check.
    """
    if start < 0:
        start += len(x)
        if start < 0:
            start = 0
    if end < 0:
        end += len(x)
        if end < 0:
            end = 0
    elif end > len(x):
        end = len(x)

    if end - start < len(prefix):
        return False
    index = start
    for i in prefix:
        if x[index] != i:
            return False
        index += 1
    return True


def bytes_endswith(x: bytes, suffix: bytes) -> bool:
    """Does given bytes object end with the subsequence suffix?

    Compiling bytes.endswith, with no range arguments, compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.
        suffix: The subsequence to look for.

    Returns:
        Result of check.
    """
    index = len(x) - len(suffix)
    if index < 0:
        return False
    for i in suffix:
        if x[index] != i:
            return False
        index += 1
    return True


def bytes_endswith_range(x: bytes, suffix: bytes, start: int, end: int) -> bool:
    """Does specified slice of given bytes object end with the subsequence suffix?

    Compiling bytes.endswith with range arguments compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.
        suffix: The subsequence to look for.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Result of check.
    """
    if start < 0:
        start += len(x)
        if start < 0:
            start = 0
    if end < 0:
        end += len(x)
        if end < 0:
            end = 0
    elif end > len(x):
        end = len(x)

    if end - start < len(suffix):
        return False
    index = end - len(suffix)
    if index < 0:
        return False
    for i in suffix:
        if x[index] != i:
            return False
        index += 1
    return True


def bytes_count(x: bytes, sub: bytes, start: int, end: int) -> int:
    """How many times does a subsequence occur within specified slice of given bytes object?

    Compiling bytes.count compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.
        sub: The subsequence to look for.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Count of non-overlapping matches.
    """
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    len_sub = len(sub)
    if len_sub == 0:
        if start > len(x):
            return 0
        count = end - start + 1
        if count < 0:
            count = 0
        return count

    count = 0
    index = start
    while index < end - len_sub + 1:
        subindex = 0
        while subindex < len_sub:
            if x[index+subindex] != sub[subindex]:
                break
            subindex += 1
            if subindex == len_sub:
                count += 1
                index += len_sub - 1
        index += 1
    return count


def bytes_count_single(x: bytes, sub: int, start: int, end: int) -> int:
    """How many times does a single byte occur within specified slice of given bytes object?

    Compiling bytes.count compiles this function, when sub is an integer 0 to 255.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object to examine.
        sub: target to search for, as a single byte specified as an integer 0 to 255.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Count of matches.

    Raises:
        ValueError: The sub argument is out of valid range.
    """
    if sub < 0 or sub > 255:
        raise ValueError("byte must be in range(0, 256)")
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    count = 0
    index = start
    while index < end:
        if x[index] == sub:
            count += 1
        index += 1
    return count


def bytes_find(x: bytes, sub: bytes, start: int, end: int) -> int:
    """Where is the first location of a subsequence within a given slice of a bytes object?

    Compiling bytes.find compiles this function, when sub is a bytes object.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object in which to search.
        sub: The subsequence to look for.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Lowest index of match within slice of x, or -1 if not found.
    """
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    len_sub = len(sub)
    if len_sub == 0:
        if start > len(x) or start > end:
            return -1
        return start

    index = start
    while index < end - len_sub + 1:
        subindex = 0
        while subindex < len_sub:
            if x[index+subindex] != sub[subindex]:
                break
            subindex += 1
            if subindex == len_sub:
                return index
        index += 1
    return -1


def bytes_find_single(x: bytes, sub: int, start: int, end: int) -> int:
    """Where is the first location of a specified byte within a given slice of a bytes object?

    Compiling bytes.find compiles this function, when sub is an integer 0 to 255.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object in which to search.
        sub: The subsequence to look for, as a single byte specified as an integer 0 to 255.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Lowest index of match within slice of x, or -1 if not found.

    Raises:
        ValueError: The sub argument is out of valid range.
    """
    if sub < 0 or sub > 255:
        raise ValueError("byte must be in range(0, 256)")
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    index = start
    while index < end:
        if x[index] == sub:
            return index
        index += 1
    return -1


def bytes_rfind(x: bytes, sub: bytes, start: int, end: int) -> int:
    """Where is the last location of a subsequence within a given slice of a bytes object?

    Compiling bytes.rfind compiles this function, when sub is a bytes object.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object in which to search.
        sub: The subsequence to look for.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
         Highest index of match within slice of x, or -1 if not found.
    """
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    len_sub = len(sub)
    if len_sub == 0:
        if start > len(x) or start > end:
            return -1
        return end

    index = end - len_sub
    while index >= start:
        subindex = 0
        while subindex < len_sub:
            if x[index+subindex] != sub[subindex]:
                break
            subindex += 1
            if subindex == len_sub:
                return index
        index -= 1
    return -1


def bytes_rfind_single(x: bytes, sub: bytes, start: int, end: int) -> int:
    """Where is the last location of a specified byte within a given slice of a bytes object?

    Compiling bytes.rfind compiles this function, when sub is an integer 0 to 255.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object in which to search.
        sub: The subsequence to look for, as a single byte specified as an integer 0 to 255.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Highest index of match within slice of x, or -1 if not found.

    Raises:
        ValueError: The sub argument is out of valid range.
    """
    if sub < 0 or sub > 255:
        raise ValueError("byte must be in range(0, 256)")
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    index = end - 1
    while index >= start:
        if x[index] == sub:
            return index
        index -= 1
    return -1


def bytes_index(x: bytes, sub: bytes, start: int, end: int) -> int:
    """Where is the first location of a subsequence within a given slice of a bytes object?

    Compiling bytes.index compiles this function, when sub is a bytes object.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object in which to search.
        sub: The subsequence to look for.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Lowest index of match within slice of x.

    Raises:
        ValueError: If sub is not found.
    """
    ret = bytes_find(x, sub, start, end)
    if ret == -1:
        raise ValueError("subsection not found")
    return ret


def bytes_index_single(x: bytes, sub: int, start: int, end: int) -> int:
    """Where is the first location of a specified byte within a given slice of a bytes object?

    Compiling bytes.index compiles this function, when sub is an integer 0 to 255.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object in which to search.
        sub: The subsequence to look for, as a single byte specified as an integer 0 to 255.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Lowest index of match within slice of x.

    Raises:
        ValueError: The sub argument is out of valid range, or sub is not found.
    """
    ret = bytes_find_single(x, sub, start, end)
    if ret == -1:
        raise ValueError("subsection not found")
    return ret


def bytes_rindex(x: bytes, sub: bytes, start: int, end: int) -> int:
    """Where is the last location of a subsequence within a given slice of a bytes object?

    Compiling bytes.rindex compiles this function, when sub is a bytes object.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object in which to search.
        sub: The subsequence to look for.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Highest index of match within slice of x.

    Raises:
        ValueError: If sub is not found.
    """
    ret = bytes_rfind(x, sub, start, end)
    if ret == -1:
        raise ValueError("subsection not found")
    return ret


def bytes_rindex_single(x: bytes, sub: int, start: int, end: int) -> int:
    """Where is the last location of a specified byte within a given slice of a bytes object?

    Compiling bytes.rindex compiles this function, when sub is an integer 0 to 255.
    This function is only intended to be executed in this compiled form.

    Args:
        x: The bytes object in which to search.
        sub: The subsequence to look for, as a single byte specified as an integer 0 to 255.
        start: Beginning of slice of x.  Interpreted as slice notation.
        end: End of slice of x.  Interpreted as slice notation.

    Returns:
        Highest index of match within slice of x.

    Raises:
        ValueError: The sub argument is out of valid range, or sub is not found.
    """
    ret = bytes_rfind_single(x, sub, start, end)
    if ret == -1:
        raise ValueError("subsection not found")
    return ret


def bytes_partition(x: bytes, sep: bytes) -> Tuple(bytes, bytes, bytes):
    """Given a bytes object and a separator, splits the object into three pieces.

    Compiling bytes.partition compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: A bytes object.
        sep: A separator.

    Returns:
        Tuple of three bytes objects:
            (1) the piece up to the first separator,
            (2) the separator,
            (3) the remainder from the first separator to the end.
    """
    if len(sep) == 0:
        raise ValueError("empty separator")

    pos = x.find(sep)
    if pos == -1:
        return Tuple(bytes, bytes, bytes)((x, b'', b''))
    return Tuple(bytes, bytes, bytes)((x[0:pos], sep, x[pos+len(sep):]))


def bytes_rpartition(x: bytes, sep: bytes) -> Tuple(bytes, bytes, bytes):
    """Given a bytes object and a separator, reverse splits the object into three pieces.

    Compiling bytes.rpartition compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: A bytes object.
        sep: A separator.

    Returns:
        Tuple of three bytes objects:
            (1) the piece from the last separator to the end,
            (2) the separator,
            (3) the remainder from the start to the last separator.
    """
    if len(sep) == 0:
        raise ValueError("empty separator")

    pos = x.rfind(sep)
    if pos == -1:
        return Tuple(bytes, bytes, bytes)((b'', b'', x))
    return Tuple(bytes, bytes, bytes)((x[0:pos], sep, x[pos+len(sep):]))


def bytes_center(x: bytes, width: int, fill: bytes) -> bytes:
    """Given a bytes object, a line width, and a fill byte, centers the bytes object within the line.

    Compiling bytes.center compiles this function.
    This function is only intended to be executed in this compiled form.
    Checking the fill length is assumed to be done outside this function.

    Args:
        x: A bytes object.
        width: Line width.
        fill: A bytes object of length 1.

    Returns:
        Transformed bytes object.
    """
    if width <= len(x):
        return x

    left = (width - len(x)) // 2
    right = (width - len(x)) - left
    return fill * left + x + fill * right


def bytes_ljust(x: bytes, width: int, fill: bytes) -> bytes:
    """Given a bytes object, a line width, and a fill byte, left-justifies the bytes object within the line.

    Compiling bytes.ljust compiles this function.
    This function is only intended to be executed in this compiled form.
    Checking the fill length is assumed to be done outside this function.

    Args:
        x: A bytes object.
        width: Line width.
        fill: A bytes object of length 1.

    Returns:
        Transformed bytes object.
    """
    if width <= len(x):
        return x

    return x + fill * (width - len(x))


def bytes_rjust(x: bytes, width: int, fill: bytes) -> bytes:
    """Given a bytes object, a line width, and a fill byte, right-justifies the bytes object within the line.

    Compiling bytes.rjust compiles this function.
    This function is only intended to be executed in this compiled form.
    Checking the fill length is assumed to be done outside this function.

    Args:
        x: A bytes object.
        width: Line width.
        fill: A bytes object of length 1.

    Returns:
        Transformed bytes object.
    """
    if width <= len(x):
        return x

    return fill * (width - len(x)) + x


def bytes_expandtabs(x: bytes, tabsize: int) -> bytes:
    """Given a bytes object and a tab size, expands all tab characters to spaces.

    Compiling bytes.expandtabs compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: A bytes object.
        tabsize: Expand tab bytes to this number of spaces.

    Returns:
        Transformed bytes object.
    """
    accumulator = ListOf(bytes)()

    col = 0  # column mod tabsize, not necessarily actual column
    last = 0
    for i in range(len(x)):
        c = x[i]
        if c == ord('\t'):
            accumulator.append(x[last:i])
            spaces = tabsize - (col % tabsize) if tabsize > 0 else 0
            accumulator.append(b' ' * spaces)
            last = i + 1
            col = 0
        elif c == ord('\n') or c == ord('\r'):
            col = 0
        else:
            col += 1
    accumulator.append(x[last:])
    return b''.join(accumulator)


def bytes_zfill(x: bytes, width: int) -> bytes:
    """Given a bytes object and width, left-pad with '0' after any initial sign character.

    Compiling bytes.zfill compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        x: A bytes object.
        width: Desired length of result.

    Returns:
        Transformed bytes object.
    """
    accumulator = ListOf(bytes)()

    sign = False
    if len(x):
        c = x[0]
        if c == ord('+') or c == ord('-'):
            accumulator.append(x[0:1])
            sign = True

    fill = width - len(x)
    if fill > 0:
        accumulator.append(b'0' * fill)

    accumulator.append(x[1:] if sign else x)

    return b''.join(accumulator)


def no_more_kwargs(context, **kwargs):
    """Helper function used when parsing arguments of compiled function calls.

    If there are any keyword arguments left, generates code that raises TypeError.

    Args:
        context: ExpressionConversionContext
        **kwargs: keyword argument dict

    """
    for e in kwargs:
        context.pushException(TypeError, f"'{e}' is an invalid keyword argument for this function")
        # just need to generate the first exception
        break


class BytesWrapper(RefcountedWrapper):
    """Code-generation wrapper for bytes type.

    Corresponds to interpreted BytesType.
    Code generated here, when compiled, is intended to behave the same as interpreted operations on BytesType.

    In all comments below, read 'bytes' as 'instance of BytesType'.
    """

    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self):
        super().__init__(bytes)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('bytecount', native_ast.Int32),
            ('data', native_ast.UInt8)
        ), name='BytesLayout').pointer()

    def getNativeLayoutType(self):
        """Returns native layout of bytes (BytesType).
        """
        return self.layoutType

    def convert_hash(self, context, expr):
        """Generates code for hash of bytes (BytesType)

        Returns:
            TypedExpression representing hash of BytesType instance.
        """
        return context.pushPod(Int32, runtime_functions.hash_bytes.call(expr.nonref_expr.cast(VoidPtr)))

    def on_refcount_zero(self, context, instance):
        """ Generates code to dispose of bytes (BytesType) instance when refcount reaches zero.

        Args:
             context: ExpressionConversionContext
             instance: Reference to bytes (BytesType) instance.

        Returns:
            native_ast code to be executed when this instance has refcount zero.
        """
        assert instance.isReference
        return runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr))

    def convert_builtin(self, f, context, expr, a1=None):
        """Generates code for builtin functions on bytes.

        Only applicable builtin is 'bytes'

        Args:
            f: builtin function
            context: ExpressionConversionContext
            expr: BytesType expression
            a1: first argument of builtin function

        Returns:
            Code for builtin function.
        """
        if f is bytes and a1 is None:
            return expr
        return super().convert_builtin(f, context, expr, a1)

    def convert_bin_op(self, context, left, op, right, inplace):
        """Generates code for bytes binary operators.

        Args:
            context: ExpressionConversionContext
            left: left operand TypedExpression of type bytes
            op: python_ast.BinaryOp operator
            right: left operand TypedExpression

        Returns:
            TypedExpression of result of operator
        """
        if op.matches.Mult and isInteger(right.expr_type.typeRepresentation):
            if left.isConstant and right.isConstant:
                return context.constant(left.constantValue * right.constantValue)

            return context.push(
                bytes,
                lambda bytesRef: bytesRef.expr.store(
                    runtime_functions.bytes_mult.call(
                        left.nonref_expr.cast(VoidPtr),
                        right.nonref_expr
                    ).cast(self.layoutType)
                )
            )

        if right.expr_type == left.expr_type:
            if op.matches.Eq or op.matches.NotEq or op.matches.Lt or op.matches.LtE or op.matches.GtE or op.matches.Gt:
                if left.isConstant and right.isConstant:
                    if op.matches.Eq:
                        return context.constant(left.constantValue == right.constantValue)
                    if op.matches.NotEq:
                        return context.constant(left.constantValue != right.constantValue)
                    if op.matches.Lt:
                        return context.constant(left.constantValue < right.constantValue)
                    if op.matches.LtE:
                        return context.constant(left.constantValue <= right.constantValue)
                    if op.matches.Gt:
                        return context.constant(left.constantValue > right.constantValue)
                    if op.matches.GtE:
                        return context.constant(left.constantValue >= right.constantValue)

                cmp_res = context.pushPod(
                    int,
                    runtime_functions.bytes_cmp.call(
                        left.nonref_expr.cast(VoidPtr),
                        right.nonref_expr.cast(VoidPtr)
                    )
                )
                if op.matches.Eq:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.eq(0)
                    )
                if op.matches.NotEq:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.neq(0)
                    )
                if op.matches.Lt:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.lt(0)
                    )
                if op.matches.LtE:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.lte(0)
                    )
                if op.matches.Gt:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.gt(0)
                    )
                if op.matches.GtE:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.gte(0)
                    )

            if op.matches.In:
                if left.isConstant and right.isConstant:
                    return context.constant(left.constantValue in right.constantValue)

                find_converted = right.convert_method_call("find", (left,), {})
                if find_converted is None:
                    return None
                return find_converted >= 0

            if op.matches.Add:
                if left.isConstant and right.isConstant:
                    return context.constant(left.constantValue + right.constantValue)

                return context.push(
                    bytes,
                    lambda bytesRef: bytesRef.expr.store(
                        runtime_functions.bytes_concat.call(
                            left.nonref_expr.cast(VoidPtr),
                            right.nonref_expr.cast(VoidPtr)
                        ).cast(self.layoutType)
                    )
                )

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_getslice(self, context, expr, lower, upper, step):
        """Generates code for bytes slice operation.

        The numeric slice arguments are interpreted as usual slice notation.
        They can be TypedExpressions of type int or any type that supports __index__.

        Args:
            context: ExpressionConversionContext
            expr: TypedExpression of type bytes being sliced
            lower: lower bound of slice, as TypedExpression.  Must be an index-able type.
            upper: upper bound of slice, as TypedExpression.  Must be an index-able type.
            step: step of slice, as TypedExpression

        Returns:
            TypedExpression of slice

        Raises:
            NotImplementedError if an unimplemented operation is attempted.
        """
        if step is not None:
            raise NotImplementedError("Slicing with a step isn't supported yet")

        if lower is None and upper is None:
            return self

        if lower is None and upper is not None:
            lower = context.constant(0)
        else:
            lower = lower.toIndex()

            if lower is None:
                return

        if lower is not None and upper is None:
            upper = expr.convert_len()

        upper = upper.toIndex()
        if upper is None:
            return

        if expr.isConstant and lower.isConstant and upper.isConstant:
            return context.constant(expr.constantValue[lower.constantValue:upper.constantValue])

        return context.push(
            bytes,
            lambda bytesRef: bytesRef.expr.store(
                runtime_functions.bytes_getslice_int64.call(
                    expr.nonref_expr.cast(native_ast.VoidPtr),
                    lower.nonref_expr,
                    upper.nonref_expr
                ).cast(self.layoutType)
            )
        )

    def convert_getitem(self, context, expr, item):
        """Generates code for bytes getitem operation.

        The index item can be a TypedExpression of type int or any type that supports __index__.

        Args:
            context: ExpressionConversionContext
            expr: TypedExpression of type bytes
            item: index of item, as TypedExpression.  Must be an index-able type.

        Returns:
            TypedExpression of indexed item
        """
        item = item.toIndex()

        if item is None:
            return None

        if expr.isConstant and item.isConstant:
            return context.constant(expr.constantValue[item.constantValue])

        len_expr = self.convert_len(context, expr)

        with context.ifelse((item.nonref_expr.lt(len_expr.nonref_expr.negate()))
                            .bitor(item.nonref_expr.gte(len_expr.nonref_expr))) as (true, false):
            with true:
                context.pushException(IndexError, "index out of range")

        return context.pushPod(
            int,
            expr.nonref_expr.ElementPtrIntegers(0, 3).elemPtr(
                native_ast.Expression.Branch(
                    cond=item.nonref_expr.lt(native_ast.const_int_expr(0)),
                    false=item.nonref_expr,
                    true=item.nonref_expr.add(len_expr.nonref_expr)
                )
            ).load().cast(native_ast.Int64)
        )

    # bytes methods that return bool, and that map to py functions
    _bool_methods = dict(
        isalnum=bytes_isalnum,
        isalpha=bytes_isalpha,
        isdigit=bytes_isdigit,
        islower=bytes_islower,
        isspace=bytes_isspace,
        istitle=bytes_istitle,
        isupper=bytes_isupper
    )

    # bytes methods with the same signature as find, and that map to py functions
    _find_methods = dict(
        count=(bytes_count, bytes_count_single),
        find=(bytes_find, bytes_find_single),
        rfind=(bytes_rfind, bytes_rfind_single),
        index=(bytes_index, bytes_index_single),
        rindex=(bytes_rindex, bytes_rindex_single),
    )

    # bytes methods that map to c++ functions
    _bytes_methods = dict(
        lower=runtime_functions.bytes_lower,
        upper=runtime_functions.bytes_upper,
        capitalize=runtime_functions.bytes_capitalize,
        swapcase=runtime_functions.bytes_swapcase,
        title=runtime_functions.bytes_title,
    )

    # list of all method names for bytes type
    _methods = ['decode', 'translate', 'maketrans', 'split', 'rsplit', 'join', 'partition', 'rpartition',
                'strip', 'rstrip', 'lstrip', 'startswith', 'endswith', 'replace',
                '__iter__', 'center', 'ljust', 'rjust', 'expandtabs', 'splitlines', 'zfill'] \
        + list(_bool_methods) + list(_bytes_methods) + list(_find_methods)

    def convert_default_initialize(self, context, target):
        context.pushEffect(
            target.expr.store(
                self.layoutType.zero()
            )
        )

    def convert_attribute(self, context, instance, attr: str):
        """Generates code for bytes attribute operation.

        Args:
            context: ExpressionConversionContext
            instance: TypedExpression of type bytes
            attr: attribute name

        Returns:
            TypedExpression of attribute value.
        """
        if attr in self._methods:
            # make attributes that are method names callable
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return super().convert_attribute(context, instance, attr)

    def has_intiter(self):
        """Does this type support the 'intiter' format?"""
        return True

    def convert_intiter_size(self, context, instance):
        """If this type supports intiter, compute the size of the iterator.

        This function will return a TypedExpression(int) or None if it set an exception."""
        return self.convert_len(context, instance)

    def convert_intiter_value(self, context, instance, valueInstance):
        """If this type supports intiter, compute the value of the iterator.

        This function will return a TypedExpression, or None if it set an exception."""
        return self.convert_getitem(context, instance, valueInstance)

    @Wrapper.unwrapOneOfAndValue
    def convert_method_call(self, context, instance, methodname: str, args, kwargs0):
        """Generates code for bytes method calls.

        Generates code raising AttributeError if methodname is invalid.
        Generates code raising TypeError if argument type is invalid.
        Generates code raising ValueError if argument value is invalid.

        Args:
            context: ExpressionConversionContext
            instance: TypedExpression of type bytes
            methodname: method name
            args: positional arguments, as tuple of TypedExpressions
            kwargs0: keyword arguments, as dict(str, TypedExpression)

        Returns:
            TypedExpression of return value of method call.
        """
        if methodname not in self._methods:
            return super().convert_method_call(context, instance, methodname, args, kwargs0)

        kwargs = kwargs0.copy()
        if methodname == '__iter__' and not args and not kwargs:
            return typeWrapper(BytesIterator).convert_type_call(
                context,
                None,
                [],
                dict(pos=context.constant(-1), bytesObj=instance)
            )

        if methodname in self._bool_methods and not args and not kwargs:
            return context.call_py_function(self._bool_methods[methodname], (instance,), {})

        if methodname in self._bytes_methods and not args and not kwargs:
            return context.push(
                bytes,
                lambda ref: ref.expr.store(
                    self._bytes_methods[methodname].call(
                        instance.nonref_expr.cast(VoidPtr)
                    ).cast(self.layoutType)
                )
            )

        if methodname in ['strip', 'lstrip', 'rstrip']:
            fromLeft = methodname in ['strip', 'lstrip']
            fromRight = methodname in ['strip', 'rstrip']
            if len(args) == 0 and not kwargs:
                return context.push(
                    bytes,
                    lambda ref: ref.expr.store(
                        runtime_functions.bytes_strip.call(
                            instance.nonref_expr.cast(VoidPtr),
                            native_ast.const_bool_expr(fromLeft),
                            native_ast.const_bool_expr(fromRight)
                        ).cast(self.layoutType)
                    )
                )
            elif len(args) == 1 and not kwargs and args[0].expr_type.typeRepresentation == bytes:
                return context.push(
                    bytes,
                    lambda ref: ref.expr.store(
                        runtime_functions.bytes_strip2.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr.cast(VoidPtr),
                            native_ast.const_bool_expr(fromLeft),
                            native_ast.const_bool_expr(fromRight)
                        ).cast(self.layoutType)
                    )
                )

        if methodname in ['startswith', 'endswith'] and not kwargs:
            if len(args) == 1:
                py_f = bytes_startswith if methodname == 'startswith' else bytes_endswith
                return context.call_py_function(py_f, (instance, args[0]), {})
            elif 2 <= len(args) <= 3:
                if len(args) == 3:
                    arg1 = args[1]
                    arg2 = args[2]
                elif len(args) == 2:
                    arg1 = args[1]
                    arg2 = self.convert_len(context, instance)
                py_f = bytes_startswith_range if methodname == 'startswith' else bytes_endswith_range
                return context.call_py_function(py_f, (instance, args[0], arg1, arg2), {})

        if methodname == 'endswith' and len(args) == 1 and not kwargs:
            return context.call_py_function(bytes_endswith, (instance, args[0]), {})
        if methodname == 'expandtabs' and len(args) == 1 and not kwargs:
            arg0type = args[0].expr_type.typeRepresentation
            if arg0type != int:
                return context.pushException(TypeError, f"an integer is required, not '{arg0type}'")
            return context.call_py_function(bytes_expandtabs, (instance, args[0]), {})

        if methodname in self._find_methods and 1 <= len(args) <= 3 and not kwargs:
            if len(args) == 3:
                start = args[1]
                end = args[2]
            elif len(args) == 2:
                start = args[1]
                end = self.convert_len(context, instance)
            elif len(args) == 1:
                start = context.constant(0)
                end = self.convert_len(context, instance)

            if isInteger(args[0].expr_type.typeRepresentation):
                py_f = self._find_methods[methodname][1]
            else:
                py_f = self._find_methods[methodname][0]
            return context.call_py_function(py_f, (instance, args[0], start, end), {})

        if methodname == 'maketrans' and len(args) == 2 and not kwargs:
            for i in [0, 1]:
                if args[i].expr_type != self:
                    context.pushException(
                        TypeError,
                        f"maketrans() argument {i + 1} must be bytes"
                    )
                    return
            arg0len = args[0].convert_len()
            arg1len = args[1].convert_len()
            if arg0len is None or arg1len is None:
                return None
            with context.ifelse(arg0len.nonref_expr.eq(arg1len.nonref_expr)) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushException(ValueError, "maketrans arguments must have same length")
            return context.push(
                bytes,
                lambda bytesRef: bytesRef.expr.store(
                    runtime_functions.bytes_maketrans.call(
                        args[0].nonref_expr.cast(VoidPtr),
                        args[1].nonref_expr.cast(VoidPtr)
                    ).cast(self.layoutType)
                )
            )

        if methodname == 'replace':
            if len(args) in [2, 3]:
                for i in [0, 1]:
                    if args[i].expr_type != self:
                        context.pushException(
                            TypeError,
                            f"replace() argument {i + 1} must be bytes"
                        )
                        return

                if len(args) == 3 and args[2].expr_type.typeRepresentation != int:
                    context.pushException(
                        TypeError,
                        f"replace() argument 3 must be int, not {args[2].expr_type.typeRepresentation}"
                    )
                    return

                arg2 = context.constant(-1) if len(args) == 2 else args[2]
                return context.call_py_function(bytes_replace, (instance, args[0], args[1], arg2), {})
                # code below is for experimenting with compiling to C++ call instead of compiled python.
                # return context.push(
                #     bytes,
                #     lambda bytesRef: bytesRef.expr.store(
                #         runtime_functions.bytes_replace.call(
                #             instance.nonref_expr.cast(VoidPtr),
                #             args[0].nonref_expr.cast(VoidPtr),
                #             args[1].nonref_expr.cast(VoidPtr),
                #             args[2].nonref_expr if len(args) == 3 else native_ast.const_int_expr(-1)
                #         ).cast(self.layoutType)
                #     )
                # )

        if methodname == 'join' and not kwargs:
            if len(args) == 1:
                # we need to pass the list of bytes objects
                separator = instance
                itemsToJoin = args[0]

                if itemsToJoin.expr_type.typeRepresentation is ListOf(bytes):
                    return context.push(
                        bytes,
                        lambda out: runtime_functions.bytes_join.call(
                            out.expr.cast(VoidPtr),
                            separator.nonref_expr.cast(VoidPtr),
                            itemsToJoin.nonref_expr.cast(VoidPtr)
                        )
                    )
                else:
                    return context.call_py_function(bytesJoinIterable, (separator, itemsToJoin), {})

        if methodname in ['split', 'rsplit'] and not kwargs:
            if len(args) == 0:
                sepPtr = VoidPtr.zero()
                maxCount = native_ast.const_int_expr(-1)
            elif len(args) in [1, 2] and args[0].expr_type.typeRepresentation in [bytes, type(None)]:
                if args[0].expr_type == typeWrapper(None):
                    sepPtr = VoidPtr.zero()
                else:
                    sepPtr = args[0].nonref_expr.cast(VoidPtr)
                    sepLen = args[0].convert_len()
                    if sepLen is None:
                        return None
                    with context.ifelse(sepLen.nonref_expr.eq(0)) as (ifTrue, ifFalse):
                        with ifTrue:
                            context.pushException(ValueError, "empty separator")

                if len(args) == 2:
                    maxCount = args[1].toInt64()
                    if maxCount is None:
                        return None
                else:
                    maxCount = native_ast.const_int_expr(-1)
            else:
                maxCount = None

            if maxCount is not None:
                fn = runtime_functions.bytes_split if methodname == 'split' else runtime_functions.bytes_rsplit
                return context.push(
                    TypedListMasqueradingAsList(ListOf(bytes)),
                    lambda outBytes: outBytes.expr.store(
                        fn.call(
                            instance.nonref_expr.cast(VoidPtr),
                            sepPtr,
                            maxCount
                        ).cast(outBytes.expr_type.getNativeLayoutType())
                    )
                )

        if methodname == 'splitlines' and not kwargs:
            if len(args) == 0:
                arg0 = context.constant(False)
            elif len(args) == 1:
                arg0 = args[0].toBool()
                if arg0 is None:
                    return None

            return context.push(
                TypedListMasqueradingAsList(ListOf(bytes)),
                lambda out: out.expr.store(
                    runtime_functions.bytes_splitlines.call(
                        instance.nonref_expr.cast(VoidPtr),
                        arg0
                    ).cast(out.expr_type.getNativeLayoutType())
                )
            )

        if methodname == 'decode' and not kwargs:
            if len(args) in [0, 1, 2]:
                return context.push(
                    str,
                    lambda ref: ref.expr.store(
                        runtime_functions.bytes_decode.call(
                            instance.nonref_expr.cast(VoidPtr),
                            (args[0] if len(args) >= 1 else context.constant(0)).nonref_expr.cast(VoidPtr),
                            (args[1] if len(args) >= 2 else context.constant(0)).nonref_expr.cast(VoidPtr),
                        ).cast(typeWrapper(str).layoutType)
                    )
                )

        if methodname == 'translate':
            if len(args) in [1, 2]:
                arg0isNone = args[0].expr_type == typeWrapper(None)
                arg0 = args[0] if not arg0isNone else context.constant(0)
                if 'delete' in kwargs and len(args) == 1:
                    arg1 = kwargs['delete']
                    del kwargs['delete']
                    no_more_kwargs(context, **kwargs)
                else:
                    arg1 = args[1] if len(args) >= 2 else context.constant(0)

                if not arg0isNone:
                    arg0type = arg0.expr_type.typeRepresentation
                    if arg0type != bytes:
                        context.pushException(TypeError, f"a bytes-like object is required, not '{arg0type}'")
                    arg0len = arg0.convert_len()
                    if arg0len is None:
                        return None
                    with context.ifelse(arg0len.nonref_expr.eq(256)) as (ifTrue, ifFalse):
                        with ifFalse:
                            context.pushException(ValueError, "translation table must be 256 characters long")

                return context.push(
                    bytes,
                    lambda ref: ref.expr.store(
                        runtime_functions.bytes_translate.call(
                            instance.nonref_expr.cast(VoidPtr),
                            arg0.nonref_expr.cast(VoidPtr),
                            arg1.nonref_expr.cast(VoidPtr),
                        ).cast(self.layoutType)
                    )
                )

        if methodname in ['partition', 'rpartition'] and len(args) == 1 and not kwargs:
            arg0type = args[0].expr_type.typeRepresentation
            if arg0type != bytes:
                context.pushException(TypeError, f"a bytes-like object is required, not '{arg0type}'")
            py_f = bytes_partition if methodname == 'partition' else bytes_rpartition
            return context.call_py_function(py_f, (instance, args[0]), {})

        if methodname in ['center', 'ljust', 'rjust']:
            if len(args) in [1, 2]:
                arg0 = args[0].toInt64()
                if arg0 is None:
                    return None

                if len(args) == 2:
                    arg1 = args[1]
                    arg1type = arg1.expr_type.typeRepresentation
                    if arg1type != bytes:
                        context.pushException(TypeError, f"{methodname}() argument 2 must be a byte string of length 1, not '{arg1type}'")
                    arg1len = arg1.convert_len()
                    if arg1len is None:
                        return None
                    with context.ifelse(arg1len.nonref_expr.eq(1)) as (ifTrue, ifFalse):
                        with ifFalse:
                            context.pushException(
                                TypeError,
                                f"{methodname}() argument 2 must be a byte string of length 1, not '{arg1type}'"
                            )
                else:
                    arg1 = context.constant(b' ')

            py_f = bytes_center if methodname == 'center' else \
                bytes_ljust if methodname == 'ljust' else \
                bytes_rjust if methodname == 'rjust' else None
            return context.call_py_function(py_f, (instance, arg0, arg1), {})

        if methodname == 'zfill' and len(args) == 1 and not kwargs:
            arg0 = args[0].toInt64()
            if arg0 is None:
                return None
            return context.call_py_function(bytes_zfill, (instance, arg0), {})

        return super().convert_method_call(context, instance, methodname, args, kwargs0)

    def convert_getitem_unsafe(self, context, expr, item):
        """Generates code for bytes getitem operation, without any argument checks.

        The index item can be a TypedExpression that can convert to int.
        Does not support __index__.

        Args:
            context: ExpressionConversionContext
            expr: TypedExpression of type bytes
            item: index of item, as TypedExpression.  Must be convertible to int.

        Returns:
            TypedExpression of item got.
        """
        return context.push(
            UInt8,
            lambda intRef: intRef.expr.store(
                expr.nonref_expr.ElementPtrIntegers(0, 3)
                    .elemPtr(item.toInt64().nonref_expr).load()
            )
        )

    def convert_len_native(self, expr):
        """Native code for bytes len operation.

        Args:
            TypedExpression of bytes type

        Returns:
            native_ast of this len operation (not a TypedExpression)
        """
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=(
                expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
            )
        )

    def convert_len(self, context, expr):
        """Generates code for bytes len operation.

        Args:
            context: ExpressionConversionContext
            expr: TypedExpression of type bytes

        Returns:
            TypedExpression representing len of bytes instance
        """
        if expr.isConstant:
            return context.constant(len(expr.constantValue))

        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))

    def constant(self, context, s: bytes):
        """Generates code for constant bytes expression

        Args:
            context: ExpressionConversionContext
            s: constant value of type bytes (really bytes, not BytesType or a TypedExpression)

        Returns:
            TypedExpression representing this constant.
        """
        return typed_python.compiler.typed_expression.TypedExpression(
            context,
            native_ast.Expression.GlobalVariable(
                name='bytes_constant_' + sha_hash(s).hexdigest,
                type=native_ast.VoidPtr,
                metadata=GlobalVariableMetadata.BytesConstant(value=s)
            ).cast(self.layoutType.pointer()),
            self,
            True,
            constantValue=s
        )

    def convert_to_type_constant(self, context, expr, target_type, level: ConversionLevel):
        """Given that 'expr' is a constant expression, attempt to convert it directly.

        This function should return None if it can't convert it to a constant, otherwise
        a typed expression with the constant.
        """
        if target_type.typeRepresentation in (str, int, float, bool):
            if level.isNewOrHigher():
                try:
                    return context.constant(target_type.typeRepresentation(expr.constantValue))
                except Exception as e:
                    context.pushException(type(e), *e.args)
                    return "FAILURE"

    def _can_convert_from_type(self, targetType, conversionLevel):
        if conversionLevel.isNewOrHigher():
            return "Maybe"

        return False

    def convert_to_self_with_target(self, context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure=False):
        if conversionLevel.isNewOrHigher():
            if sourceVal.expr_type.typeRepresentation in (ListOf(UInt8), TupleOf(UInt8)):
                # we have a fastpath for this
                context.pushEffect(
                    targetVal.expr.store(
                        runtime_functions.list_or_tuple_of_to_bytes.call(
                            sourceVal.nonref_expr.cast(native_ast.VoidPtr),
                            context.getTypePointer(sourceVal.expr_type.typeRepresentation)
                        ).cast(targetVal.expr_type.layoutType)
                    )
                )

                return context.constant(True)

            # note - need a better way of determining whether we just want to let the interpreter
            # handle this.
            if issubclass(sourceVal.expr_type.typeRepresentation, (ListOf, TupleOf, Dict, ConstDict, Set)):
                return context.call_py_function(
                    convertIterableToBytes,
                    (targetVal.asPointer(), sourceVal, context.constant(mayThrowOnFailure)),
                    {}
                )

        return super().convert_to_self_with_target(context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure)

    def _can_convert_to_type(self, targetType, conversionLevel):
        if not conversionLevel.isNewOrHigher():
            return False

        if targetType.typeRepresentation in (bytes, float, int, bool):
            return True

        if targetType.typeRepresentation is str:
            return "Maybe"

        return False

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        if targetVal.expr_type.typeRepresentation is bool:
            res = context.pushPod(bool, self.convert_len_native(instance.nonref_expr).neq(0))
            context.pushEffect(
                targetVal.expr.store(res.nonref_expr)
            )
            return context.constant(True)

        if targetVal.expr_type.typeRepresentation is int:
            res = context.pushPod(
                int,
                runtime_functions.bytes_to_int64.call(instance.nonref_expr.cast(VoidPtr))
            )
            context.pushEffect(
                targetVal.expr.store(res.nonref_expr)
            )
            return context.constant(True)

        if targetVal.expr_type.typeRepresentation is float:
            res = context.pushPod(
                float,
                runtime_functions.bytes_to_float64.call(instance.nonref_expr.cast(VoidPtr))
            )
            context.pushEffect(
                targetVal.expr.store(res.nonref_expr)
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def get_iteration_expressions(self, context, expr):
        """Generates fixed list of iteration values of the bytes expression.

        Possible for constant bytes expressions only.

        Args:
            context: ExpressionConversionContext
            expr: TypedExpression of type bytes to be iterated

        Returns:
            list of TypedExpressions unrolling iteration of this bytes expression
        """
        if expr.isConstant:
            return [context.constant(expr.constantValue[i]) for i in range(len(expr.constantValue))]
        else:
            return None


class BytesIterator(Class, Final):
    pos = Member(int)
    bytesObj = Member(bytes)
    value = Member(int)

    def __fastnext__(self):
        self.pos = self.pos + 1

        if self.pos < len(self.bytesObj):
            self.value = self.bytesObj[self.pos]
            return pointerTo(self).value
        else:
            return PointerTo(int)()


class BytesMaketransWrapper(Wrapper):
    """Code-generation wrapper for the bytes.maketrans static builtin function.

    Code generated here, when compiled, is intended to behave the same as bytes.maketrans in the interpreter.

    In all comments below, read 'bytes' as 'instance of BytesType'.
    """
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(bytes.maketrans)

    def getNativeLayoutType(self):
        """bytes.maketrans has no layout.
        """
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        """Generate code for calling the bytes.maketrans static builtin function.

        Args:
            expr: TypedExpression of bytes.maketrans type
            args: positional arguments of bytes.maketrans
            kwargs: keyword arguments of bytes.maketrans (N/A)

        Returns:
            TypedExpression representing return value of bytes.maketrans
        """
        if len(args) == 2 and not kwargs:
            return args[0].convert_method_call("maketrans", (args[0], args[1]), {})

        return super().convert_call(context, expr, args, kwargs)
