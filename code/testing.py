"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Applies the testing cycle on a dataloader.
    It also gets testing metrics.
"""

import segmentation_models_pytorch as smp
import pandas as pd


def test_report(model, dataloader, criterion, metrics, device, report_path, save_name):
    """
        test_report Runs testing and creates a CSV report.

        @param model Model to use for testing.
        @param dataloader Dataloader with testing images.
        @param criterion Loss criterion.
        @param metrics Metrics to know from the training phases.
        @param device Use of GPU.
        @param report_path Path to store the results report.
        @param save_name Name to save the report with.
    """

    test_epoch = smp.utils.train.ValidEpoch(\
        model=model,\
        loss=criterion,\
        metrics=metrics,\
        device=device)

    # Test model
    metrics = test_epoch.run(dataloader)

    # Create metrics dataframe
    metrics_df = pd.DataFrame(metrics, index=[0])

    # Save final dataframe
    metrics_df.to_csv(report_path + save_name + '.csv')
