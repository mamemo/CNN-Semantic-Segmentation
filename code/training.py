"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to implement the training and validation cycles.
"""

import torch
import segmentation_models_pytorch as smp


def train_validate(model, train_loader, val_loader, optimizer,\
                    criterion, metrics, device, epochs, save_criteria,\
                    weights_path, save_name):
    """ 
        train_validate Trains and validates a model.

        @param model Model to train on.
        @param train_loader Images to train with.
        @param val_loader Images to use for validation.
        @param optimizer Optimizer to update weights.
        @param criterion Loss criterion.
        @param metrics Metrics to know from the training phases.
        @param device Use of GPU.
        @param epochs Amount of epochs to train.
        @param save_criteria What metric to use to save best weights.
        @param weights_path Path to the folder to save best weights.
        @param save_name Filename of the best weights.
    """

    # Initial best model values
    best_criteria = 0
    best_model = {}

    train_epoch = smp.utils.train.TrainEpoch(\
        model,\
        loss=criterion,\
        metrics=metrics,\
        optimizer=optimizer,\
        device=device,\
        verbose=True)

    valid_epoch = smp.utils.train.ValidEpoch(\
        model,\
        loss=criterion,\
        metrics=metrics,\
        device=device,\
        verbose=True)

    # Iterates over total epochs
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}')
        # Train
        metrics = train_epoch.run(train_loader)
        # Validate
        if val_loader:
            metrics = valid_epoch.run(val_loader)

        # Update best model
        if epoch == 1 or metrics[save_criteria] >= best_criteria:
            best_criteria = metrics[save_criteria]
            best_model = {'epoch': epoch,\
                'model_state_dict': model.state_dict(),\
                'optimizer_state_dict': optimizer.state_dict(),\
                'dice_loss': metrics['dice_loss'],\
                'iou_score': metrics['iou_score'],\
                'fscore': metrics['fscore'],\
                'accuracy': metrics['accuracy']}

    # Save model
    save_path = '{}{}_{}_{:.6}.pth'.format(weights_path, save_name,\
                save_criteria, str(best_criteria).replace('.', '_'))
    torch.save(best_model, save_path)

    return save_path
