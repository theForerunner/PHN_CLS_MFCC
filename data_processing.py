import os
import math
import json
from pathlib import Path

import numpy as np

from tqdm.contrib.concurrent import process_map

import librosa

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

from loguru import logger

def collate_fn_pad(data):
    """collate_fn_pad
    Pads batch of variable length

    :param batch:
    """

    spect, times, segmentation_labels, phonemes, phoneme_labels, wav_paths = data[0]

    return torch.FloatTensor(spect).cuda(), times, torch.LongTensor(segmentation_labels).cuda(), phonemes, torch.FloatTensor(phoneme_labels).cuda(), wav_paths

def print_list(list_to_print, line_width=25):
    for idx, item in enumerate(list_to_print):
        print(f'{item:2d}', end=", ")
        if (idx + 1) % line_width == 0:
            print()
    if len(list_to_print) % line_width != 0:
        print()

class TimitDictionary:
    def __init__(self, cfg):
        reduction_dict_path = cfg.phn_reduction_dict_path
        idx_dict_path = cfg.phn_idx_dict_path

        with open(reduction_dict_path, 'r') as fp:
            self.phn_reduction_dict = json.load(fp)

        self.phonemes = set(self.phn_reduction_dict.values())
        self.n_phonemes = len(self.phonemes)

        with open(idx_dict_path, 'r') as fp:
            self.phoneme2idx = json.load(fp)

        self.idx2phoneme = {v:k for k,v in self.phoneme2idx.items()}

    def phn2idx(self, phoneme):
        phoneme = self.phn_reduction_dict[phoneme]
        return self.phoneme2idx[phoneme]

    def phns2idx(self, phonemes):
        return [self.phn2idx(phoneme) for phoneme in phonemes]

class TimitDataset(Dataset):
    def __init__(self, data_path, cfg):
        super(TimitDataset, self).__init__()

        self.wav_paths = [os.path.join(data_path, f) for f in Path(data_path).rglob('*.WAV')]

        self.cfg = cfg

        self.vocab = TimitDictionary(cfg)

        if cfg.dev_run:
            dev_run_size = cfg.dev_run_size
            if dev_run_size < 1:
                dev_run_size = math.ceil(dev_run_size * len(self.wav_paths))
            self.wav_paths = self.wav_paths[:dev_run_size]

        logger.info(f'Loading {len(self.wav_paths)} files into cache ...')
        self.cache = process_map(self.process_file, self.wav_paths, max_workers=cfg.num_workers, chunksize=5)

    @staticmethod
    def get_train_dataset(data_path, cfg):
        train_dataset = TimitDataset(os.path.join(data_path, 'train'), cfg)

        train_len   = len(train_dataset)
        train_split = int(train_len * (1 - cfg.val_ratio))
        val_split   = train_len - train_split
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_split, val_split])

        return train_dataset, val_dataset

    @staticmethod
    def get_test_dataset(data_path, cfg):
        return TimitDataset(os.path.join(data_path, 'test'), cfg)

    def process_file(self, wav_path):

        def pad2seq_and_mfcc():
            spect_seq = []
            n_chunk = math.ceil(audio_len / chunk_size)
            for i in range(1, n_chunk + 1):
                if i < buffer_size:
                    spect = F.pad(audio[:i * chunk_size], [chunk_size * (buffer_size - i), 0])
                elif i == n_chunk:
                    spect = F.pad(audio[(i - buffer_size) * chunk_size:], [0, chunk_size - audio_len % chunk_size if audio_len % chunk_size else 0])
                else:
                    spect = audio[(i - buffer_size) * chunk_size:i * chunk_size]
                spect_mfcc = librosa.feature.mfcc(spect.numpy(), sr, n_mfcc=self.cfg.n_mfcc, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, n_mels=self.cfg.n_mels)
                # spect_mfcc_weighted = spect_mfcc * np.arange(0.1, 1, 0.1)
                # spect_seq.append(spect_mfcc_weighted[np.newaxis, :])
                spect_seq.append(spect_mfcc[np.newaxis, :])
            return np.array(spect_seq)

        def segmentation_to_labels():
            labels = np.zeros(n_chunk, dtype=np.int64)
            
            for seg in segmentation_times[:-1]:
                labels[math.ceil(seg / chunk_size) - 1] = 1
            
            return labels

        def phoneme_to_labels():
            labels = np.zeros((n_chunk, self.vocab.n_phonemes), dtype=np.float32)
            phoneme_indexes = self.vocab.phns2idx(phonemes)
            for i in range(n_chunk):
                no_class_flag = True
                for j, seg in enumerate(segmentation_times):
                    seg_start_pos = math.ceil(segmentation_times[j - 1] / chunk_size) - 1 if j > 0 else 0
                    seg_end_pos = math.ceil(seg / chunk_size) - 1
                    if seg_start_pos <= i <= seg_end_pos:
                        labels[i][phoneme_indexes[j]] = 1.0
                        no_class_flag = False
                if no_class_flag:
                    labels[i][10] = 1.0 # Set to silence if no phoneme type given

            return labels

        phn_path = wav_path.replace("WAV", "PHN")

        # load audio
        audio, sr = torchaudio.load(wav_path, normalize=False)
        audio = audio[0]

        # Normalize data to [-1, 1]
        audio = 2 * (audio - torch.min(audio)) / (torch.max(audio) - torch.min(audio)) - 1.0 

        audio_len = len(audio)
        chunk_size = self.cfg.chunk_size
        buffer_size = self.cfg.buffer_size
        n_chunk = math.ceil(audio_len / chunk_size)

        # Generate spect sequence with buffer and get MFCC
        spect_seq = pad2seq_and_mfcc()

        # Load labels -- segmentation and phonemes
        with open(phn_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            lines = list(map(lambda line: line.split(" "), lines))

            # Get segment times
            segmentation_times = list(map(lambda line: int(line[1]), lines))

            # Get segment label
            segmentation_labels = segmentation_to_labels()

            # Get phonemes in each segment (for K times there should be K+1 phonemes)
            phonemes = list(map(lambda line: line[2].strip(), lines))
            phoneme_labels = phoneme_to_labels()

        return spect_seq, segmentation_times, segmentation_labels, phonemes, phoneme_labels, wav_path

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.wav_paths)