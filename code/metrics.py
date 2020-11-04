"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Implementation of metrics and results accumulator.

    * Supported Metrics and Accumulators
        - Loss
        - Dice
        - IoU
        - Pixel Wise Accuracy

"""


import numpy as np
import torch
from torch import nn


class Metrics():
    """
        Class to calculate metrics over training and testing.
    """

    def __init__(self):
        """
            init Constructor method.

            @param self Created object.
        """

        # * Accumulators for Metrics
        self.loss = 0
        self.dice = 0
        self.iou = 0
        self.accuracy = 0
        self.batches = 0
        
        self.activation_fn = nn.Sigmoid()


    def batch(self, labels, preds, loss=0):
        """
            batch Method to update metrics acummulators for every batch of
                    processed images.

            @param self Object.
            @param labels Respective labels of the processed images.
            @param preds Predicted outputs.
            @param loss Loss of predictions over labels.
        """
        self.dice += self.diceCoeff(gt=labels, pred=preds)
        self.iou += self.miou(labels=labels, preds=preds)
        self.accuracy += self.pixel_accuracy(labels=labels, preds=preds)

        self.loss += loss
        self.batches += 1


    def summary(self):
        """
            summary Returns a summary of the metrics results.

            @param self Object.
        """
        return {
            "Model Dice":           [self.dice / self.batches],
            "Model IoU":            [self.iou / self.batches],
            "Model Pixel Accuracy": [self.accuracy / self.batches]
            "Model Loss":           [self.loss]
        }


    def print_summary(self):
        """
            print_summary Prints the summary of the metrics results.

            @param self Object.
        """
        summ = self.summary()
        for key in summ:
            print(key+": ", summ[key])


    def print_one_liner(self, phase='Train'):
        """
            print_one_liner Prints a quick summary of the metrics results in one line.

            @param self Object.
            @param phase Respective run phase where the function is called from.
        """
        summ = self.summary()
        print('{} Dice: {:.4}, {} IoU: {:.4}, {} Acc: {:.4}, {} Loss: {:.4} '\
            .format(phase, summ["Model Dice"][0], phase, summ["Model IoU"][0], \
                    phase, summ["Model Pixel Accuracy"][0], phase, summ["Model Loss"][0]))
        return summ


    def diceCoeff(self, gt, pred, eps=1e-5):
        """
            diceCoeff Measures the Dice Coefficient of the predictions.

            @param self Object.
            @param gt Annotations that come as groundtruth.
            @param pred Predictions made by the model.
            @param eps Epsilon to avoid undetermined division.
        """

        pred = self.activation_fn(pred)
    
        N = gt.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
    
        tp = torch.sum(gt_flat * pred_flat, dim=1)
        fp = torch.sum(pred_flat, dim=1) - tp
        fn = torch.sum(gt_flat, dim=1) - tp
        loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        return loss.sum() / N


    def miou(self, labels, preds, eps=1e-5):
        """
            miou Measures the mean Intersection over Union of the predictions.

            @param self Object.
            @param labels Annotations that come as groundtruth.
            @param preds Predictions made by the model.
            @param eps Epsilon to avoid undetermined division.
        """

        preds = self.activation_fn(preds)

        intersection = (preds & labels).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
        union = (preds | labels).float().sum((2, 3))

        result = (intersection + eps * (union == 0)) / (union - intersection + eps)

        return result.mean()


    def pixel_accuracy(self, labels, preds):
        """
            pixel_accuracy Measures the Pixel wise Accuracy of the predictions.

            @param self Object.
            @param labels Annotations that come as groundtruth.
            @param preds Predictions made by the model.
        """

        tmp = preds == labels

        return (torch.sum(tmp).float() / preds.nelement())
