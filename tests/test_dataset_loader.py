from unittest.mock import MagicMock, patch

from src.dataset_loader import CNNDailyMailDatasetLoader, DatasetBatch


class TestCNNDailyMailDatasetLoader:
    @patch("src.dataset_loader.load_dataset")
    def test_load_returns_dataset_batch(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda s: 5
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = [
            {"article": "Article one text.", "highlights": "Summary one."},
            {"article": "Article two text.", "highlights": "Summary two."},
            {"article": "Article three text.", "highlights": "Summary three."},
        ]
        mock_ds.__iter__ = lambda s: iter(s.select())
        mock_load.return_value = mock_ds

        loader = CNNDailyMailDatasetLoader()
        batch = loader.load(split="train", limit=3, shuffle=True, seed=42)

        assert isinstance(batch, DatasetBatch)
        assert len(batch.texts) == 3
        assert len(batch.references) == 3
        assert batch.texts[0] == "Article one text."
        assert batch.references[0] == "Summary one."

    @patch("src.dataset_loader.load_dataset")
    def test_load_respects_limit(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda s: 100
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = [
            {"article": f"Article {i}.", "highlights": f"Summary {i}."}
            for i in range(5)
        ]
        mock_ds.__iter__ = lambda s: iter(s.select())
        mock_load.return_value = mock_ds

        loader = CNNDailyMailDatasetLoader()
        batch = loader.load(limit=5)

        mock_ds.select.assert_called_once()
        assert len(batch.texts) == 5

    @patch("src.dataset_loader.load_dataset")
    def test_load_without_shuffle(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda s: 3
        mock_ds.select.return_value = [
            {"article": "Text.", "highlights": "Summary."},
        ]
        mock_ds.__iter__ = lambda s: iter(s.select())
        mock_load.return_value = mock_ds

        loader = CNNDailyMailDatasetLoader()
        loader.load(shuffle=False, limit=1)

        mock_ds.shuffle.assert_not_called()

    @patch("src.dataset_loader.load_dataset")
    def test_load_without_limit(self, mock_load):
        rows = [
            {"article": f"Text {i}.", "highlights": f"Sum {i}."}
            for i in range(3)
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda s: 3
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.__iter__ = lambda s: iter(rows)
        mock_load.return_value = mock_ds

        loader = CNNDailyMailDatasetLoader()
        batch = loader.load(limit=None)

        mock_ds.select.assert_not_called()
        assert len(batch.texts) == 3

    @patch("src.dataset_loader.load_dataset")
    def test_version_passed_to_load_dataset(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda s: 1
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = [
            {"article": "Text.", "highlights": "Sum."},
        ]
        mock_ds.__iter__ = lambda s: iter(s.select())
        mock_load.return_value = mock_ds

        loader = CNNDailyMailDatasetLoader(version="3.0.0")
        loader.load(split="test", limit=1)

        mock_load.assert_called_once_with("cnn_dailymail", "3.0.0", split="test")
