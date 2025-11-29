def loss_function(ignore_index):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)

def get_optim(parameters, lr):
    return torch.optim.Adam(parameters=parameters, lr=lr)
