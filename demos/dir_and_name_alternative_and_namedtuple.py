"""
Alex bug about using dir and name for Alternatives
"""

from typed_python import Alternative, OneOf

ColorDefinition = Alternative(
    "ColorDefinition",
    Rgb=dict(
        red=float,
        blue=float,
        green=float
    ),
    Named=dict(
        name=OneOf('red', 'blue', 'white', 'black', 'green', 'yellow')
    )
)

rgb = ColorDefinition.Rgb(red=1.0, blue=.5, green=0.0)

import pdb; pdb.set_trace()