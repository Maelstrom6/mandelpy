import unittest
import mandelpy.settings as s
from numba.cuda.compiler import DeviceFunctionTemplate


class TestSettings(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.settings = s.Settings()

    def tearDown(self) -> None:
        pass

    def test_functions(self):
        """
        Makes sure the getter and setter methods are working correctly
        """

        self.assertIsInstance(self.settings.fn, DeviceFunctionTemplate)
        self.assertIsInstance(self.settings.transform, DeviceFunctionTemplate)
        self.assertIsInstance(self.settings.inv_transform, DeviceFunctionTemplate)

        self.settings.fn = None
        self.assertIsInstance(self.settings.fn, DeviceFunctionTemplate)

        self.settings.fn = lambda z, c: z ** 3 + c
        self.assertIsInstance(self.settings.fn, DeviceFunctionTemplate)

        self.settings.transform = None
        self.assertIsInstance(self.settings.transform, DeviceFunctionTemplate)

        self.settings.transform = lambda z, c: z ** 3 + c
        self.assertIsInstance(self.settings.transform, DeviceFunctionTemplate)

        self.settings.inv_transform = None
        self.assertIsInstance(self.settings.inv_transform, DeviceFunctionTemplate)

        self.settings.inv_transform = lambda z, c: z ** 3 + c
        self.assertIsInstance(self.settings.inv_transform, DeviceFunctionTemplate)

    def test_focal(self):

        self.settings.focal = (0, 0, 4)
        self.assertEqual(self.settings.left, -4)
        self.assertEqual(self.settings.right, 4)
        self.assertEqual(self.settings.top, 4)
        self.assertEqual(self.settings.bottom, -4)


if __name__ == '__main__':
    unittest.main()
