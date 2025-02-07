from neural_networks.base_score_model import BaseScoreModel
from neural_networks.basic_blocks.adpative_normalization import SelectiveAdaptiveNormalization
from neural_networks.basic_blocks.mlp import MLP, MLPInject
from neural_networks.model_informed import ModelInformationLikelihood
from utils.config import ScoreModelType


def get_score_neural_network(in_n, in_m, in_k, score_model_type,
                             in_problem,*args,
                             **kwargs):
    if score_model_type in [ScoreModelType.PriorLikelihoodSplitIID,
                            ScoreModelType.PriorLikelihoodSplitIIDModelInformed]:
        in_data_size = in_n
    else:
        in_data_size = in_n * in_k
    data_dim =  in_data_size * (int(in_problem.is_complex) + 1) # if complex, double the size of the data
    theta_dim = in_m
    prior_model = generate_prior_mlp(theta_dim, kwargs)
    likelihood_model = generate_likelihood_mlp(data_dim, theta_dim, in_problem, score_model_type,
                                                                     kwargs)
    return BaseScoreModel(theta_dim,
                          prior_model,
                          likelihood_model,
                          score_model_type,
                          theta_injection=kwargs["inject"],
                          theta_project=None,
                          is_complex=in_problem.is_complex)


def generate_likelihood_mlp(in_data_size, theta_dim, in_problem, score_model_type, kwargs):
    def normalization():
        return SelectiveAdaptiveNormalization(1, in_problem.get_condition_list())

    if score_model_type == ScoreModelType.PriorLikelihoodSplitIIDModelInformed:
        mi = in_problem.get_model_information()
        likelihood_model = MLPInject(in_data_size + mi.output_size(), mi.output_size(), mi.output_size(),
                                     normalization=normalization,
                                     n_layers=kwargs["n_layers"], feature_size=kwargs["feature_size"],
                                     se_block=kwargs["se_block"], non_linearity=kwargs["non_linearity"],
                                     non_linearity_normalization=kwargs["non_linearity_normalization"],
                                     transformer=False, bypass=False, inject=kwargs["inject"],
                                     bias_output=kwargs["output_bias"],
                                     output_rescale=kwargs["output_rescale"])
        likelihood_model = ModelInformationLikelihood(mi, likelihood_model)
    else:
        likelihood_model = MLPInject(in_data_size + theta_dim, theta_dim, theta_dim,
                                     normalization=normalization,
                                     n_layers=kwargs["n_layers"], feature_size=kwargs["feature_size"],
                                     se_block=kwargs["se_block"], non_linearity=kwargs["non_linearity"],
                                     non_linearity_normalization=kwargs["non_linearity_normalization"],
                                     transformer=False, bypass=False, inject=kwargs["inject"],
                                     bias_output=kwargs["output_bias"],
                                     output_rescale=kwargs["output_rescale"])
    return likelihood_model


def generate_prior_mlp(in_m, kwargs):
    prior_model = MLP(in_m, in_m, bias_output=kwargs["output_bias_prior"], normalization=None,
                      n_layers=kwargs["n_layers_prior"], feature_size=kwargs["feature_size_prior"],
                      se_block=kwargs["se_block_prior"],
                      non_linearity=kwargs["non_linearity_prior"])
    return prior_model
