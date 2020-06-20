"""
Main testing module for generator.py

Source:
https://www.youtube.com/watch?v=6tNS--WetLI
"""
import unittest
import mandelpy.generator as g


class TestGenerator(unittest.TestCase):

    def test_identify_blocks(self):
        result = g.identify_blocks(500, 500, False, False)
        desired = [
            (0, 0, 500, 500)
        ]
        self.assertEqual(result, desired)

        result = g.identify_blocks(500, 500, False, False, block_size=(250, 250))
        desired = [
            (0, 0, 250, 250),
            (0, 250, 250, 250),
            (250, 0, 250, 250),
            (250, 250, 250, 250)
        ]
        self.assertEqual(set(result), set(desired))

        result = g.identify_blocks(1100, 1100, False, False, block_size=(500, 500))
        desired = [
            (0, 0, 500, 500),
            (0, 500, 500, 500),
            (0, 1000, 500, 100),
            (500, 0, 500, 500),
            (500, 500, 500, 500),
            (500, 1000, 500, 100),
            (1000, 0, 100, 500),
            (1000, 500, 100, 500),
            (1000, 1000, 100, 100)
        ]
        self.assertEqual(set(result), set(desired))

        result = g.identify_blocks(10, 10, True, False, block_size=(500, 500))
        desired = [
            (0, 0, 10, 5)
        ]
        self.assertEqual(set(result), set(desired))

        # with self.assertRaises(ValueError):
        #     g.identify_blocks(10, 10)

        # self.assertRaises(ValueError, g.identify_blocks, 10, 10)

    def test_compile_kernel(self):
        from mandelpy.settings import Settings
        from mandelpy.kernels import factories
        settings = Settings()
        args = (
            settings.width, settings.height,
            settings.left, settings.right,
            settings.top, settings.bottom,
            settings.max_iter, settings.threshold,
            settings.z0, settings.fn,
            settings.transform, settings.inv_transform,
            settings.orbit_id
        )

        result = g.compile_kernel(settings)
        desired = factories["mand"](*args)
        self.assertEqual(result.py_func.__name__, desired.py_func.__name__)

        settings.tipe = "dbwaonge"
        result = g.compile_kernel(settings)
        desired = factories["mand"](*args)
        self.assertEqual(result.py_func.__name__, desired.py_func.__name__)

        settings.tipe = "buddha"
        result = g.compile_kernel(settings)
        desired = factories["buddha"](*args)
        self.assertEqual(result.py_func.__name__, desired.py_func.__name__)


if __name__ == '__main__':
    unittest.main()
