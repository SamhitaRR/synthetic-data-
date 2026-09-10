"""
multi_channel_predictor.py
===========================
Subclass of nnUNetPredictor that routes export calls to
export_multichannel_prediction_from_logits, writing 3 separate
output files per case instead of one.

Export runs synchronously to avoid background workers hanging on
large image resampling.

Copy to:
    nnunetv2/inference/multi_channel_predictor.py

Usage:
    python -m nnunetv2.inference.multi_channel_predictor \
        -i /path/to/input \
        -o /path/to/output \
        -d 048 \
        -c 3d_fullres \
        -f 0 \
        -chk checkpoint_best.pth
"""

import os
from time import sleep

import numpy as np
import torch

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.multi_channel_export import export_multichannel_prediction_from_logits
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache


class MultiChannelPredictor(nnUNetPredictor):
    """
    Identical to nnUNetPredictor except:
    - export uses export_multichannel_prediction_from_logits (3 output files)
    - export runs synchronously in the main process (no background pool)
      to avoid hanging on large image resampling
    """

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = 1):
        ret = []

        for preprocessed in data_iterator:
            data = preprocessed['data']
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed['ofile']
            if ofile is not None:
                print(f'\nPredicting {os.path.basename(ofile)}:')
            else:
                print(f'\nPredicting image of shape {data.shape}:')

            print(f'perform_everything_on_device: {self.perform_everything_on_device}')

            properties = preprocessed['data_properties']

            prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()

            if ofile is not None:
                print('resampling and exporting...')
                export_multichannel_prediction_from_logits(
                    prediction, properties, self.configuration_manager,
                    self.plans_manager, self.dataset_json, ofile, save_probabilities
                )
                ret.append(None)
                print(f'done with {os.path.basename(ofile)}')
            else:
                result = convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction, self.plans_manager, self.configuration_manager,
                    self.label_manager, properties, save_probabilities
                )
                ret.append(result)
                print(f'\nDone with image of shape {data.shape}:')

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret


if __name__ == '__main__':
    import argparse
    from nnunetv2.paths import nnUNet_results

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',   type=str, required=True,  help='Input folder')
    parser.add_argument('-o',   type=str, required=True,  help='Output folder')
    parser.add_argument('-d',   type=str, required=True,  help='Dataset ID e.g. 048')
    parser.add_argument('-c',   type=str, required=True,  help='Configuration e.g. 3d_fullres')
    parser.add_argument('-f',   type=int, default=0,      help='Fold')
    parser.add_argument('-tr',  type=str, default='nnUNetTrainerMultiChannelSeg', help='Trainer name')
    parser.add_argument('-p',   type=str, default='nnUNetPlans', help='Plans name')
    parser.add_argument('-chk', type=str, default='checkpoint_final.pth', help='Checkpoint filename')
    parser.add_argument('--save_probabilities', action='store_true')
    args = parser.parse_args()

    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    dataset_name = maybe_convert_to_dataset_name(args.d)
    model_folder = os.path.join(nnUNet_results, dataset_name,
                                f'{args.tr}__{args.p}__{args.c}')

    predictor = MultiChannelPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(args.f,),
        checkpoint_name=args.chk
    )
    predictor.predict_from_files(
        args.i, args.o,
        save_probabilities=args.save_probabilities,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=1
    )
