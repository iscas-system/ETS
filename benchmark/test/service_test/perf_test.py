from django.test import TestCase
import requests


class PerfTest(TestCase):
    def test_hello(self):
        url = '127.0.0.1'
        data = {'model': 'resnet50', 'batch_size': 2, 'input_size': 224, 'dtype': 'float'}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=data, headers=headers)
        self.assertEqual(response.status_code, 200)
