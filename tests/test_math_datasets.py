import unittest
import importlib
from unittest import mock

import datasets

math_datasets = importlib.import_module("pipelinerl.domains.math.load_datasets")


class MathDatasetLoaderTest(unittest.TestCase):
    def test_load_json_dataset_fetches_remote_json_via_requests(self):
        payload = [[{"value": "Solve 1+1"}, {"ground_truth": {"value": "2"}}]]

        response = mock.Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None

        with mock.patch.object(math_datasets.requests, "get", return_value=response) as get_mock:
            dataset = math_datasets._load_json_dataset("https://example.com/data.json")

        self.assertIsInstance(dataset, datasets.Dataset)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["0"]["value"], "Solve 1+1")
        get_mock.assert_called_once_with("https://example.com/data.json", timeout=300)


if __name__ == "__main__":
    unittest.main()
