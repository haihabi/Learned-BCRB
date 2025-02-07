import torch


class ScoreResultAnalyzer:

    def __init__(self, per_condition=False):
        self.results = {}
        self.per_condition = per_condition

    def get_condition_analyzer(self):
        base = self

        class ConditionAnalyzer:
            def __init__(self):
                self.error_norm = []
                self.norm_ref = []
                self.condition = []
                self.delta = []
                self.ref = []

            def add_result(self, in_score_learned, in_score_optimal, in_condition):
                self.error_norm.append(torch.linalg.norm(in_score_learned - in_score_optimal, dim=-1))
                self.norm_ref.append(torch.linalg.norm(in_score_optimal, dim=-1))
                self.condition.append(in_condition)
                self.delta.append(in_score_learned - in_score_optimal)
                self.ref.append( in_score_optimal)

            def analyze(self, name):
                result = {}
                delta = torch.cat(self.delta, dim=0)
                ref = torch.cat(self.ref, dim=0)
                error_norm_array = torch.cat(self.error_norm, dim=0)
                norm_ref_array = torch.cat(self.norm_ref, dim=0)
                condition_array = torch.cat(self.condition)

                result["re_" + name] = 100 * torch.mean(error_norm_array).item() / torch.mean(norm_ref_array).item()
                result["score_norm2_" + name] = torch.mean(error_norm_array ** 2).item()
                result["score_bias_" + name] = torch.linalg.norm(torch.mean(delta,dim=0)).item()/torch.mean(torch.linalg.norm(ref,dim=-1),dim=0).item()
                # result["score_bias_" + name] = torch.mean(error_norm_array).item()
                if base.per_condition:
                    for c in condition_array.unique():
                        mask = condition_array == c
                        result[f"cond_{str(c.item())}" + "/re_" + name] = 100 * torch.mean(
                            error_norm_array[mask]).item() / torch.mean(
                            norm_ref_array[mask]).item()
                        result[f"cond_{str(c.item())}" + "/score_norm2_" + name] = torch.mean(  # noqa
                            error_norm_array[mask] ** 2).item()
                        result[f"cond_{str(c.item())}" + "/score_bias_" + name] = torch.linalg.norm(torch.mean(delta[mask],dim=0)).item()/torch.mean(torch.linalg.norm(ref[mask],dim=-1),dim=0).item()
                        # result[f"cond_{str(c.item())}" + "/score_bias_ratio_" + name] = torch.linalg.norm(
                        #     torch.mean(delta[mask], dim=0)+torch.mean(ref, dim=0)).item() / torch.linalg.norm(torch.mean(ref, dim=0)).item()
                return result

        return ConditionAnalyzer()

    def add_result(self, in_score_name, in_score_learned, in_score_optimal, in_condition):
        if in_score_name not in self.results:
            self.results[in_score_name] = self.get_condition_analyzer()
        self.results[in_score_name].add_result(in_score_learned, in_score_optimal, in_condition)

    def analyze(self):
        results = {}
        for score_name, analyzer in self.results.items():
            results.update(analyzer.analyze(score_name))
        return results

    def clear(self):
        self.results = {}
