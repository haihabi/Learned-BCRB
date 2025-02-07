from neural_networks.basic_blocks.non_linearity_factory import get_non_linearity
from neural_networks.score_model_factory import get_score_neural_network
from signal_models.signals_model_factory import get_signal_model


def param2score_nn(device, param, problem):
    non_linearity = get_non_linearity(param.non_linearity)
    non_linearity_prior = get_non_linearity(param.non_linearity_prior)
    model = get_score_neural_network(in_n=param.n,
                                     in_k=param.k,
                                     in_m=param.m,
                                     score_model_type=param.score_model_type,
                                     n_layers=param.n_layers,
                                     feature_size=param.feature_size,
                                     se_block=param.se_block,
                                     non_linearity=non_linearity,
                                     n_layers_prior=param.n_layers_prior,
                                     feature_size_prior=param.feature_size_prior,
                                     se_block_prior=param.se_block_prior,
                                     non_linearity_prior=non_linearity_prior,
                                     in_problem=problem,
                                     output_bias_prior=param.output_bias_prior,
                                     non_linearity_normalization=param.non_linearity_normalization,
                                     output_bias=param.output_bias,
                                     inject=param.inject,
                                     output_rescale=param.output_rescale).to(device)

    return model


def param2problem(param):
    problem = get_signal_model(param.signal_model,
                               n=param.n,
                               m=param.m,
                               k=param.k,
                               alpha_rho=param.alpha_rho,
                               beta_rho=param.beta_rho,
                               sigma_prior=param.sigma_prior,
                               rho_cov=param.rho_cov,
                               n_bits=param.n_bits,
                               threshold=param.threshold,
                               noise_type=param.noise_type,
                               spacing=param.spacing,
                               is_random_phase=param.is_random_phase,
                               minimal_snr=param.minimal_snr,
                               maximal_snr=param.maximal_snr
                               )  # Get the problem
    return problem
