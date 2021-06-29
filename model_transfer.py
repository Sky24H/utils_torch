import torch

def model_transfer(model, checkpoint_path, save_name=None):
    # save as the same filename if not specified.
    if save_name == None:
        save_name = checkpoint_path

    # load model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # save model
    torch.save(model.state_dict(), save_name, _use_new_zipfile_serialization=False)


# checkpoint = 'models/vgg16-00b39a1b.pth'
# model, layerList = modelSelector(checkpoint)
# model_transfer(model, checkpoint)