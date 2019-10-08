#   Copyright 2017-2019 typed_python Authors
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


def indent(text, amount=4, ch=' '):
    padding = amount * ch
    return ''.join(padding+line for line in text.splitlines(True))


def distance(s1, s2):
    """Compute the edit distance between s1 and s2"""
    if len(s1) < len(s2):
        return distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j+1]+1, cur[j]+1, prev[j] + (1 if c1 != c2 else 0)))
        prev = cur

    return prev[-1]


def closest_in(name, names):
    return sorted((distance(name, x), x) for x in names)[0][1]


def closest_N_in(name, names, count):
    return [x[1] for x in sorted((distance(name, x), x) for x in names)[:count]]
