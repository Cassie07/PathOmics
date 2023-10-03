def extracted_feature(patch_loader, model):
    model.eval()

    for i, data in enumerate(patch_loader_loader):
        with torch.no_grad():
            images, image_names = data
            images = images.to(device)
            image_features = model(images)
    return image_features, image_names