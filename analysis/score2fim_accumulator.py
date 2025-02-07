import torch


class Score2FIMAccumulator:
    def __init__(self, split_iid_in_eval=True, rescale_iid=1, remove_mean=False):
        self.fim_list = []
        self.score_mean_list = []
        self.size_list = []
        self.c_max = 0
        self.remove_mean = remove_mean
        self.rescale_iid = rescale_iid

    def __len__(self):
        return len(self.fim_list)

    def get_fim(self):
        """
        Get the Fisher Information Matrix.
        :return: The Fisher Information Matrix.
        """
        if len(self.fim_list) == 0:
            return None
        size_list = torch.tensor(self.size_list, dtype=torch.float32, device=self.fim_list[0].device)
        w = size_list / size_list.sum()  # Weighted average to account for different batch sizes
        fim = torch.sum(torch.stack(self.fim_list) * w.reshape([-1, 1, 1]), dim=0)
        fim = fim
        if self.remove_mean:  # Remove mean
            score_mean = torch.stack(self.score_mean_list)
            score_mean = torch.sum(score_mean * w.reshape([-1, 1]), dim=0)
            fim = fim - score_mean.unsqueeze(dim=-1) @ score_mean.unsqueeze(dim=-2)
        return fim / self.rescale_iid

    def accumulate(self, score):
        """
        Accumulate Score to Fisher Information Matrix.
        :param score: The score.
        :return: None
        """
        with torch.no_grad():
            if self.remove_mean:
                self.score_mean_list.append(score.mean(dim=(0, 1) if score.dim() == 3 else 0, keepdim=False))
            fim = score.unsqueeze(dim=-1) @ score.unsqueeze(dim=-2)
            iter_fim = torch.mean(fim,
                                  dim=(0, 1) if score.dim() == 3 else 0)
            c_max = torch.linalg.norm(torch.mean(fim, dim=1) if score.dim() == 3 else fim, dim=(1, 2), ord=2).max()
            self.c_max = max(self.c_max, c_max)

            self.fim_list.append(iter_fim.detach())
            self.size_list.append(score.shape[0])

    def clear(self):
        """
        Clear the accumulator.
        :return: None
        """
        self.fim_list = []
        self.size_list = []


cond2item = lambda x: x.item() if isinstance(x, torch.Tensor) else x


class ConditionedScore2FIMAccumulator:
    def __init__(self, condition_list):
        self.condition_list = condition_list
        self.accumulators = {
            cond2item(condition): Score2FIMAccumulator() for condition
            in condition_list}

    def get_fim(self) -> torch.Tensor:
        """
        Get the Fisher Information Matrix.
        :return: The Fisher Information Matrix.
        """
        fim_list = [self.accumulators[cond2item(condition)].get_fim()
                    for condition in self.condition_list]
        return torch.stack(fim_list)

    def accumulate(self, score: torch.Tensor, condition):
        """
        Accumulate Score to Fisher Information Matrix.
        :param score: The score.
        :param condition: The condition.
        :return: None
        """
        for c in self.condition_list:
            self.accumulators[cond2item(c)].accumulate(score[c == condition, :])

    def clear(self):
        """
        Clear the accumulator.
        :return: None
        """
        for condition in self.condition_list:
            self.accumulators[cond2item(condition)].clear()
