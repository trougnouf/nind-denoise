
# requires Sigmoid
def confident_mse_loss(answer, target):
    not_confident_i = ((answer > 0.45) & (answer < 0.55)).to(torch.float32)
    confident_i = 1-not_confident_i
    dif = torch.abs(answer-target)
    res = confident_i*dif**2+not_confident_i*dif
    return res.mean()
