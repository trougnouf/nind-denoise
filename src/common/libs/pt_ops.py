def pt_crop_batch(batch, cs: int):
    '''
    center crop an image batch to cs
    also compatible with numpy tensors
    '''
    x0 = (batch.shape[3]-cs)//2
    y0 = (batch.shape[2]-cs)//2
    return batch[:, :, y0:y0+cs, x0:x0+cs]

def crop_to_multiple(tensor, multiple=64):
    return tensor[...,:tensor.size(-2)-tensor.size(-2)%multiple,:tensor.size(-1)-tensor.size(-1)%multiple]
