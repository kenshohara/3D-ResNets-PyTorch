def get_fine_tuning_parameters(model, ft_module_names):
    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break

    return parameters