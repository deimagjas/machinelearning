from unittest.mock import MagicMock, patch

from pipeline.train import Metrics, run_training


class TestMetrics:
    def test_on_train_loss_report(self):
        m = Metrics()
        m.on_train_loss_report({"iteration": 10, "train_loss": 2.5})
        m.on_train_loss_report({"iteration": 20, "train_loss": 1.8})
        assert m.train_losses == [(10, 2.5), (20, 1.8)]

    def test_on_val_loss_report(self):
        m = Metrics()
        m.on_val_loss_report({"iteration": 50, "val_loss": 0.7})
        assert m.val_losses == [(50, 0.7)]

    def test_initial_state(self):
        m = Metrics()
        assert m.train_losses == []
        assert m.val_losses == []


class TestRunTraining:
    @patch("pipeline.train.train")
    @patch("pipeline.train.CacheDataset")
    @patch("pipeline.train.TrainingArgs")
    @patch("pipeline.train.optim.Adam")
    def test_calls_train_with_correct_args(
        self, mock_adam, mock_training_args, mock_cache, mock_train
    ):
        model = MagicMock()
        mock_cache.side_effect = lambda x: x
        mock_args_instance = MagicMock()
        mock_training_args.return_value = mock_args_instance
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        metrics = run_training(
            model=model,
            train_set=["a", "b"],
            val_set=["c"],
            training_args_dict={"batch_size": 1, "iters": 200, "steps_per_eval": 50},
            adapter_file="adapters/adapters.safetensors",
            learning_rate=1e-5,
        )

        assert mock_cache.call_count == 2
        mock_training_args.assert_called_once_with(
            adapter_file="adapters/adapters.safetensors",
            batch_size=1,
            iters=200,
            steps_per_eval=50,
        )
        mock_adam.assert_called_once_with(learning_rate=1e-5)
        mock_train.assert_called_once()
        call_kwargs = mock_train.call_args
        assert call_kwargs[0][0] is model
        assert call_kwargs[0][1] is mock_optimizer
        assert call_kwargs.kwargs["args"] is mock_args_instance
        assert isinstance(metrics, Metrics)
