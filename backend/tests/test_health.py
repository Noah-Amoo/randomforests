import unittest

from app import app


class HealthTests(unittest.TestCase):
    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})
        self.assertEqual(response.mimetype, "application/json")

    def test_local_frontend_origin(self) -> None:
        response = self.client.get(
            "/health", headers={"Origin": "http://localhost:5173"}
        )
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Origin"),
            "http://localhost:5173",
        )

    def test_other_origin_is_not_allowed(self) -> None:
        response = self.client.get(
            "/health", headers={"Origin": "https://example.com"}
        )
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))


if __name__ == "__main__":
    unittest.main()
