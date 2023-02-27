import unittest
from app import *

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_post(self):
        response = self.app.post(Genre=Action, Type=TV, Producer=Bandai Visual, Studio=Sunrise, Popularity=1000, Members=1000, Episodes=20, Source=Original)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()