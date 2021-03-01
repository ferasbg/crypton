import sys
import os
import time
import math
import torch
import argparse
import h5py
import json

"""
Define conventional/nominal, certification/verification (robustness, safety), mpc, adversarial evaluation metrics for network.

Recorded metrics:

Nominal/Conventional Accuracy Trained With Geforce K80:
1563/1563 [==============================] - 9s 6ms/step - loss: 0.1419 - accuracy: 0.9522 - val_loss: 1.2321 - val_accuracy: 0.7442

Certified Accuracy Trained With Geforce K80:
1563/1563 [==============================] - {}s {}ms/step - certified_loss: 0.1419 - certified_accuracy: {} - certified_val_loss: {} - certified_val_accuracy: {}



"""

class Metrics():
    @staticmethod
    def nominal_cross_entropy_loss(self):
        """ Compute Cross-Entropy Loss Given compute_softmax(Model model) """
        raise NotImplementedError

    @staticmethod
    def mean_accuracy(self, correct_pixels, total_pixels):
        """Mean Pixel Accuracy. Proportion of all correctly classified pixels over the total pixels of the image, then compute sum of all proportions, then average it out.
        """
        raise NotImplementedError

    @staticmethod
    def certified_robustness_accuracy(prediction_robustness_threshold, certified_accuracy):
        """robustSize(scores, , δ, L) returns the certified robustness size, which is then compared to the prediction robustness threshold T."""
        certified_robustness_size = {}
        if (certified_robustness_size > prediction_robustness_threshold):
            return certified_robustness_size

    @staticmethod
    def nominal_kl_divergence():
        raise NotImplementedError

    @staticmethod
    def get_certification_accuracy():
        raise NotImplementedError

    @staticmethod
    def k_anonymity():
        """
        "this approach was proposed by Sweeney in 2002 [6]. A dataset is said to be k-anonymous if
        every combination of identity-revealing characteristics occurs in at least k different rows of the dataset.
        This anonymization approach is vulnerable to such attacks as background knowledge attacks."
        """
        raise NotImplementedError

    @staticmethod
    def l_diversity():
        """
        "it was proposed by Machanavajjhala et al. in 2007 [7]. The l-diversity scheme was proposed
        to handle some weaknesses in the k-anonymity scheme by promoting intra-group diversity of sensitive
        data within the anonymization scheme [8]. It is prone to skewness and similarity attacks" 
        """
        raise NotImplementedError

    @staticmethod
    def t_closeness():
        '''
        "this anonymization scheme was proposed by Li et al. in 2007 [9]. It is a refinement
        of l-diversity discussed above [8]. It requires that distribution of sensitive attributes within each
        quasi-identifier group should be “close” to their distribution in the entire original dataset (that is, the
        distance between the two distributions should be no more than a threshold t)"
        '''
        raise NotImplementedError

    