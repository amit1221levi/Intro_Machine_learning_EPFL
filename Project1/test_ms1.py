"""
Module for auto-testing student projects.
This is based on the file from Francois Fleuret's
"Deep Learning Course": https://fleuret.org/dlc/.

This is the Milestone 1 version.
"""

import re
import sys
import os
import unittest
import importlib
from pathlib import Path

import numpy as np


class HidePrints:
    """Disable normal printing for calling student code."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class NoHidePrints:
    """Don't disable normal printing for calling student code."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestProject(unittest.TestCase):

    @staticmethod
    def title(msg):
        print(f"\n==============\n> {msg} ...")
    

    def test_1_folder_structure(self):
        """Test the framework structure (folder and files)."""
        self.title("Testing folder structure")
        self.assertTrue(project_path.exists(), f"No folder found at {project_path}")

        # Main files
        for file in ["main.py", "report.pdf"]:
            with self.subTest(f"Checking file {file}"):
                self.assertTrue((project_path / file).exists(), f"No file {file} found at {project_path}")
        
        # Source code
        src_path = project_path / "src"
        self.assertTrue(src_path.exists(), f"{src_path} not found")
        for file in ["__init__.py", "data.py", "utils.py"]:
            with self.subTest(f"Checking file src/{file}"):
                self.assertTrue((src_path / file).exists(), f"No file {file} found at {src_path}")
        # Methods
        method_path = src_path / "methods"
        self.assertTrue(method_path.exists(), f"{method_path} not found")
        for file in ["__init__.py", "dummy_methods.py",
                     "kmeans.py", "logistic_regression.py", "svm.py"]:
            with self.subTest(f"Checking file methods/{file}"):
                self.assertTrue((method_path / file).exists(), f"No file {file} found at {method_path}")

    
    def _import_and_test(self, name, class_name, *args, **kwargs):
        """Test the import of the method and its functions."""
        # Code structure
        module = importlib.import_module(f"src.methods.{name}")
        method = module.__getattribute__(class_name)(*args, **kwargs)
        for fn in ["fit", "predict"]:
            _ = method.__getattribute__(fn)
        if name == "kmeans":
            _ = method.__getattribute__("k_means")
        
        # Functions inputs and outputs
        N, D = 10, 3
        training_data = np.random.rand(N, D)
        training_labels = np.random.randint(0, D, N)
        test_data = np.random.rand(N, D)
        with no_print():
            pred_labels = method.fit(training_data, training_labels)
        self.assertIsInstance(pred_labels, np.ndarray, f"{name}.{class_name}.fit() should output an array, not {type(pred_labels)}")
        self.assertEqual(pred_labels.shape, training_labels.shape, f"{name}.{class_name}.fit() output has wrong shape ({pred_labels.shape} != {training_labels.shape})")
        with no_print():
            pred_labels = method.predict(test_data)
        self.assertIsInstance(pred_labels, np.ndarray, f"{name}.{class_name}.predict() should output an array, not {type(pred_labels)}")
        self.assertEqual(pred_labels.shape, training_labels.shape, f"{name}.{class_name}.predict() output has wrong shape ({pred_labels.shape} != {training_labels.shape})")
        
        return method


    def test_2_dummy_methods(self):
        """Test the dummy methods."""
        self.title("Testing dummy methods")

        _ = self._import_and_test("dummy_methods", "DummyClassifier",
                                  arg1=1)
    

    def test_3a_kmeans(self):
        """Test K-Means."""
        self.title("Testing KMeans")

        kmeans_model = self._import_and_test("kmeans", "KMeans",
                                             K=2)

        # Test on easy dummy data
        training_data = np.array([[0., 0.], [1., 0.], [0., 1.], [5., 5.]])
        training_labels = np.array([0, 0, 0, 1])
        test_data = np.array([[0.5, 0.5], [-10., -10.], [5.1, 5.1], [10., 5.]])
        test_labels = np.array([0, 0, 1, 1])
        with no_print():
            pred_labels_train = kmeans_model.fit(training_data, training_labels)
            pred_labels_test = kmeans_model.predict(test_data)
        self.assertTrue(np.equal(pred_labels_train, training_labels).all(), f"KMeans.fit() is not working on dummy data")
        self.assertTrue(np.equal(pred_labels_test, test_labels).all(), f"KMeans.predict() is not working on dummy data")
    

    def test_3b_logistic_regression(self):
        """Test Logistic Regression."""
        self.title("Testing logistic regression")

        logistic_regression = self._import_and_test("logistic_regression", "LogisticRegression",
                                                    lr=1e-3, max_iters=500)

        # Test on easy dummy data
        N = 20
        training_data = np.concatenate([
            np.linspace(-5, -0.25, N//2)[:,None],
            np.linspace(0.25, 5, N//2)[:,None]
        ], axis=0)
        training_labels = (training_data[:,0] > 0.).astype(int)
        test_data = np.array([-10., -5., -1., 1., 5., 10.])[:, None]
        test_labels = (test_data[:,0] > 0.).astype(int)
        with no_print():
            pred_labels_train = logistic_regression.fit(training_data, training_labels)
            pred_labels_test = logistic_regression.predict(test_data)
        self.assertTrue((pred_labels_train == training_labels).all(), f"LogisticRegression.fit() is not working on dummy data")
        self.assertTrue((pred_labels_test == test_labels).all(), f"LogisticRegression.predict() is not working on dummy data")
        

    def test_3c_svm(self):
        """Test SVM."""
        self.title("Testing SVM")

        svm_model = self._import_and_test("svm", "SVM",
                                          C=10.0, kernel="rbf", gamma=1)

        # Test on easy dummy data
        N = 10
        training_data = np.linspace(0, 2 * np.pi, N)
        training_data = np.stack([np.cos(training_data), np.sin(training_data)], axis=1)
        training_data = np.concatenate([training_data, np.array([[-0.2, 0.2], [0.2, 0.2], [-0.2, -0.2], [0.2, -0.2]])])
        training_labels = np.array([0] * N + [1] * 4)
        test_data = np.array([[0., 0.], [1, 0], [0, 2], [-0.7, -0.7]]).astype(float)
        test_labels = np.array([1, 0, 0, 0])
        with no_print():
            pred_labels_train = svm_model.fit(training_data, training_labels)
            pred_labels_test = svm_model.predict(test_data)
        self.assertTrue(np.isclose(pred_labels_train, training_labels).all(), f"SVM.fit() is not working on dummy data")
        self.assertTrue(np.isclose(pred_labels_test, test_labels).all(), f"SVM.predict() is not working on dummy data")


def warn(msg):
    print(f"\33[33m/!\\ Warning: {msg}\33[39m")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--no-hide', action='store_true', help='Enable printing from the student code')
    args = parser.parse_args()
    
    project_path = Path(".")

    dir_name = project_path.absolute().name
    if re.match(r'^((\d{6})_){3}project$', dir_name) is None:
        warn("Project folder name must be in the form 'XXXXXX_XXXXXX_XXXXXX_project'")

    if args.no_hide:
        no_print = NoHidePrints
    else:
        no_print = HidePrints

    unittest.main(argv=[''], verbosity=0)