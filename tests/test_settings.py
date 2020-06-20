import unittest
import mandelpy.settings as s
from numba.cuda.compiler import DeviceFunctionTemplate


class TestSettings(unittest.TestCase):

    def test_functions(self):
        """
        Makes sure the getter and setter methods are working correctly
        """
        settings = s.Settings()
        self.assertIsInstance(settings.fn, DeviceFunctionTemplate)
        self.assertIsInstance(settings.transform, DeviceFunctionTemplate)
        self.assertIsInstance(settings.inv_transform, DeviceFunctionTemplate)

        settings.fn = None
        self.assertIsInstance(settings.fn, DeviceFunctionTemplate)

        settings.fn = lambda z, c: z ** 3 + c
        self.assertIsInstance(settings.fn, DeviceFunctionTemplate)

        settings.transform = None
        self.assertIsInstance(settings.transform, DeviceFunctionTemplate)

        settings.transform = lambda z, c: z ** 3 + c
        self.assertIsInstance(settings.transform, DeviceFunctionTemplate)

        settings.inv_transform = None
        self.assertIsInstance(settings.inv_transform, DeviceFunctionTemplate)

        settings.inv_transform = lambda z, c: z ** 3 + c
        self.assertIsInstance(settings.inv_transform, DeviceFunctionTemplate)

    def test_focal(self):
        settings = s.Settings()

        settings.focal = (0, 0, 4)
        self.assertEqual(settings.left, -4)
        self.assertEqual(settings.right, 4)
        self.assertEqual(settings.top, 4)
        self.assertEqual(settings.bottom, -4)


if __name__ == '__main__':
    unittest.main()
