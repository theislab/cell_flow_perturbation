from typing import Any

from flax.training import train_state


class BNTrainState(train_state.TrainState):
    batch_stats: Any
