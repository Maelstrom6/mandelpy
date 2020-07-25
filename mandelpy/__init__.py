"""
The mandelpy package.

Examples:
    To create a slideshow of all the presets:

    >>> from mandelpy.settings import presets
    >>> for key, setting in presets
    ...     img = generator.create_image(setting, verbose=False)
    ...     img.show()


"""

from .settings import Settings, presets
from .validators import power
from .generator import create_image, Generator
