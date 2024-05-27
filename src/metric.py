import torch
from torchmetrics import Metric


class MRMetric(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: True

    def __init__(self):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += ranks.sum()

    def compute(self):
        return self.rank_sum / self.total
    

class MRRMetric(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: True

    def __init__(self):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += (1./ ranks).sum()

    def compute(self):
        return self.rank_sum / self.total
    

class HitsMetric(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: True

    def __init__(self, topk=1):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.topk = topk

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += (ranks <= self.topk).sum()

    def compute(self):
        return self.rank_sum / self.total
    
