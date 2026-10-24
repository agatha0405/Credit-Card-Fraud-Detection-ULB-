from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import app


class TransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.threshold, cls.features = app.load_artifacts()
        cls.frame = pd.read_csv(app.DEMO_PATH, float_precision="round_trip")

    def read_frame(self, frame):
        return app.read_transactions(frame.to_csv(index=False).encode("utf-8"), self.features)

    def test_model_cache_reuses_resource(self):
        self.assertIs(app.load_artifacts()[0], self.model)

    def test_missing_model_files(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "best_model.joblib"):
                app.load_artifacts(Path(folder))

    def test_predictions_match_saved_model_and_threshold(self):
        parsed = self.read_frame(self.frame)
        result = app.predict_transactions(parsed, self.model, self.features, self.threshold)
        expected = self.model.predict_proba(parsed[self.features])[:, 1]
        np.testing.assert_allclose(result["fraud_score"], expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result["prediction"], expected >= self.threshold)
        pd.testing.assert_frame_equal(result[self.frame.columns], parsed)

    def test_unlabeled_csv_and_reversed_columns(self):
        frame = self.frame.drop(columns="Class").loc[:, self.features[::-1]]
        parsed = self.read_frame(frame)
        with patch.object(self.model, "predict_proba", wraps=self.model.predict_proba) as predict:
            result = app.predict_transactions(parsed, self.model, self.features, self.threshold)
        model_input = predict.call_args.args[0]
        self.assertEqual(model_input.columns.tolist(), self.features)
        pd.testing.assert_frame_equal(model_input, parsed[self.features])
        self.assertNotIn("Class", model_input)
        self.assertEqual(app.evaluate_labels(result), (None, None))

    def test_download_round_trip_preserves_rows_and_results(self):
        result = app.predict_transactions(self.frame, self.model, self.features, self.threshold)
        downloaded = pd.read_csv(StringIO(result.to_csv(index=False)), float_precision="round_trip")
        pd.testing.assert_frame_equal(downloaded, result, check_exact=True)
        self.assertEqual(downloaded.columns.tolist(), [*self.frame.columns, "fraud_score", "prediction"])

    def test_missing_feature(self):
        with self.assertRaisesRegex(ValueError, "Thiếu cột.*V1"):
            self.read_frame(self.frame.drop(columns="V1"))

    def test_extra_id_preserves_leading_zeros(self):
        frame = self.frame.assign(transaction_id=[f"{number:04d}" for number in range(len(self.frame))])
        parsed = self.read_frame(frame)
        self.assertEqual(parsed["transaction_id"].tolist(), frame["transaction_id"].tolist())
        result = app.predict_transactions(parsed, self.model, self.features, self.threshold)
        exported = pd.read_csv(StringIO(result.to_csv(index=False)), dtype={"transaction_id": "string"})
        self.assertEqual(exported["transaction_id"].tolist(), frame["transaction_id"].tolist())

    def test_non_numeric_feature(self):
        frame = self.frame.assign(Amount="không phải số")
        with self.assertRaisesRegex(ValueError, "dữ liệu số.*Amount"):
            self.read_frame(frame)

    def test_missing_value(self):
        frame = self.frame.copy()
        frame.loc[0, "V1"] = np.nan
        with self.assertRaisesRegex(ValueError, "giá trị thiếu"):
            self.read_frame(frame)

    def test_infinite_value(self):
        frame = self.frame.copy()
        frame.loc[0, "V1"] = np.inf
        with self.assertRaisesRegex(ValueError, "vô hạn"):
            self.read_frame(frame)

    def test_duplicate_feature_header(self):
        frame = pd.concat([self.frame, self.frame[["Time"]]], axis=1)
        with self.assertRaisesRegex(ValueError, "Tên cột bị trùng.*Time"):
            self.read_frame(frame)

    def test_malformed_csv_row(self):
        text = self.frame.to_csv(index=False).splitlines()
        text[1] += ",extra"
        with self.assertRaisesRegex(ValueError, "số cột không khớp"):
            app.read_transactions("\n".join(text).encode("utf-8"), self.features)

    def test_empty_csv(self):
        for payload in [b"", self.frame.head(0).to_csv(index=False).encode("utf-8")]:
            with self.subTest(payload=payload[:20]):
                with self.assertRaises(ValueError):
                    app.read_transactions(payload, self.features)

    def test_reserved_output_columns(self):
        with self.assertRaisesRegex(ValueError, "đã có cột kết quả"):
            self.read_frame(self.frame.assign(fraud_score=0))

    def test_valid_labels_match_sklearn_metrics(self):
        result = app.predict_transactions(self.frame, self.model, self.features, self.threshold)
        evaluation, message = app.evaluate_labels(result)
        self.assertIsNone(message)
        self.assertEqual(evaluation["matrix"].sum(), len(result))
        self.assertAlmostEqual(evaluation["metrics"]["Recall"], app.recall_score(result["Class"], result["prediction"]))

    def test_invalid_labels_do_not_block_predictions(self):
        for labels in ["không rõ", np.nan, 2]:
            with self.subTest(labels=labels):
                frame = self.frame.assign(Class=labels)
                parsed = self.read_frame(frame)
                result = app.predict_transactions(parsed, self.model, self.features, self.threshold)
                evaluation, message = app.evaluate_labels(result)
                self.assertIsNone(evaluation)
                self.assertIn("Class không hợp lệ", message)
                self.assertEqual(len(result), len(frame))

    def test_single_class_skips_undefined_evaluation(self):
        frame = self.frame.loc[self.frame["Class"].eq(0)]
        result = app.predict_transactions(frame, self.model, self.features, self.threshold)
        evaluation, message = app.evaluate_labels(result)
        self.assertIsNone(evaluation)
        self.assertIn("một lớp", message)


class DashboardTests(unittest.TestCase):
    def test_empty_dashboard_and_demo_filters(self):
        dashboard = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(dashboard.exception), 0)
        self.assertTrue(dashboard.button[0].disabled)
        dashboard.radio[0].set_value("Dùng dữ liệu demo").run()
        dashboard.button[0].click().run()
        self.assertEqual(len(dashboard.exception), 0)
        self.assertEqual(len(dashboard.dataframe[0].value), 20)
        self.assertGreaterEqual(len(dashboard.get("download_button")), 2)
        dashboard.selectbox[0].set_value("Nghi gian lận").run()
        table = dashboard.dataframe[0].value
        self.assertTrue(table["Kết quả"].eq("Nghi gian lận").all())
        self.assertTrue(table["fraud_score"].is_monotonic_decreasing)
        dashboard.selectbox[0].set_value("Tất cả giao dịch").run()
        dashboard.checkbox[0].set_value(False).run()
        self.assertEqual(dashboard.dataframe[0].value["Dòng dữ liệu"].tolist(), list(range(1, 21)))
        dashboard.radio[0].set_value("Tải CSV").run()
        self.assertEqual(len(dashboard.dataframe), 0)
        self.assertEqual(len(dashboard.exception), 0)

    def test_uploaded_csv_with_and_without_labels(self):
        real_frame = pd.read_csv(app.DEMO_PATH)
        for with_labels in [True, False]:
            with self.subTest(with_labels=with_labels):
                frame = real_frame if with_labels else real_frame.drop(columns="Class")
                payload = frame.to_csv(index=False).encode("utf-8")
                uploaded = SimpleNamespace(name="transactions.csv", getvalue=lambda: payload)
                with patch("streamlit.file_uploader", return_value=uploaded):
                    dashboard = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=30).run()
                    dashboard.button[0].click().run()
                self.assertEqual(len(dashboard.exception), 0)
                self.assertEqual(len(dashboard.error), 0)
                self.assertEqual(len(dashboard.dataframe[0].value), len(frame))
                self.assertEqual(len(dashboard.metric), 8 if with_labels else 5)

    def test_invalid_upload_shows_error_without_crashing(self):
        payload = pd.read_csv(app.DEMO_PATH).drop(columns="Amount").to_csv(index=False).encode("utf-8")
        uploaded = SimpleNamespace(name="invalid.csv", getvalue=lambda: payload)
        with patch("streamlit.file_uploader", return_value=uploaded):
            dashboard = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=30).run()
            dashboard.button[0].click().run()
        self.assertEqual(len(dashboard.exception), 0)
        self.assertIn("Thiếu cột đầu vào: Amount", dashboard.error[0].value)
        self.assertEqual(len(dashboard.dataframe), 0)

    def test_missing_artifact_shows_startup_error(self):
        with patch("app.load_artifacts", side_effect=FileNotFoundError("Thiếu best_model.joblib")):
            dashboard = AppTest.from_string("import app\napp.main()", default_timeout=30).run()
        self.assertEqual(len(dashboard.exception), 0)
        self.assertIn("Thiếu best_model.joblib", dashboard.error[0].value)


if __name__ == "__main__":
    unittest.main()
