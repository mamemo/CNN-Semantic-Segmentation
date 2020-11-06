"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to run testing metrics once the model has trained.
"""


import glob
import pathlib

import torch
import segmentation_models_pytorch as smp

from hyperparameters import parameters as params
from models import create_model
from dataset import get_dataloader
from testing import test_report
from focal_loss import FocalLoss


def main():
    """
        main Main function, flow of program.
    """

    # Model
    model = create_model(model=params['model'], encoder=params['encoder'],\
                        encoder_weights=params['encoder_weights'])

    # Running architecture (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using GPU?: ', torch.cuda.is_available())

    # Image loader
    proc_fn = smp.encoders.get_preprocessing_fn(params['encoder'], params['encoder_weights'])
    test_loader = get_dataloader(data_file=params['test_file'], img_size=params['img_size'],\
            batch_size=1, proc_fn=proc_fn, data_split='Testing')

    # Creates the criterion (loss function)
    criterion = smp.utils.losses.DiceLoss()

    # Weights Load Up
    weights_file = glob.glob(params['weights_path']+'/'+params['save_name']+'*.pth')[0]

    checkpoint = torch.load(weights_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model Loaded!\nDice Loss: {:.4}\nIoU: {:.4}\nFscore: {:.4}\nAccuracy: {:.4}'\
            .format(checkpoint['dice_loss'], checkpoint['iou_score'],\
                    checkpoint['fscore'], checkpoint['accuracy']))


    # Create folder for weights
    pathlib.Path(params['report_path']).mkdir(parents=True, exist_ok=True)

    # Metrics
    metrics = [\
        smp.utils.metrics.IoU(threshold=0.5),\
        smp.utils.metrics.Fscore(threshold=0.5),\
        smp.utils.metrics.Accuracy(threshold=0.5),\
        smp.utils.metrics.Recall(threshold=0.5),\
        smp.utils.metrics.Precision(threshold=0.5)]

    # Run test metrics and creates a report
    test_report(model=model, dataloader=test_loader, criterion=criterion,\
                metrics=metrics, device=device, report_path=params['report_path'],\
                save_name=params['save_name'])


if __name__ == "__main__":
    main()
