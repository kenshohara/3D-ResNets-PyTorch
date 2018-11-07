def get_mean_std(value_scale, dataset):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        mean = [0.4339, 0.4046, 0.3776]
        std = [0.1512, 0.1486, 0.1570]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std