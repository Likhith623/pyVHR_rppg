#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Downloads FaceForensics++ and Deep Fake Detection public data release.
See: https://github.com/ondyari/FaceForensics

Example usage (for Neuro-Pulse project):
  # Download 50 real videos
  python download_FaceForensics.py ./ff_downloads --dataset original --compression c40 --type videos --num_videos 50 --server EU2

  # Download 50 fake videos
  python download_FaceForensics.py ./ff_downloads --dataset Deepfakes --compression c40 --type videos --num_videos 50 --server EU2
"""

import argparse
import os
import urllib
import urllib.request
import tempfile
import time
import sys
import json
from tqdm import tqdm
from os.path import join

# --- URLs and filenames ---
FILELIST_URL = 'misc/filelist.json'
DEEPFAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5']

# --- Parameters ---
DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures',
}
ALL_DATASETS = [
    'original', 'DeepFakeDetection_original', 'Deepfakes',
    'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures',
]
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics++ public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('output_path', type=str, help='Output directory.')
    parser.add_argument(
        '-d', '--dataset', type=str, default='all',
        choices=list(DATASETS.keys()) + ['all'],
        help='Which dataset to download.',
    )
    parser.add_argument(
        '-c', '--compression', type=str, default='raw',
        choices=COMPRESSION,
        help='Compression level. c40 = smallest (recommended for Neuro-Pulse).',
    )
    parser.add_argument(
        '-t', '--type', type=str, default='videos',
        choices=TYPE,
        help='File type: videos, masks, or models.',
    )
    parser.add_argument(
        '-n', '--num_videos', type=int, default=None,
        help='Number of videos to download (leave blank for all).',
    )
    parser.add_argument(
        '--server', type=str, default='EU2',
        choices=SERVERS,
        help='Download server. EU2 is the only active server currently.',
    )
    args = parser.parse_args()

    if args.server == 'EU':
        server_url = 'http://canis.vc.in.tum.de:8100/'
    elif args.server == 'EU2':
        server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'
    elif args.server == 'CA':
        server_url = 'http://falas.cmpt.sfu.ca:8100/'
    else:
        raise ValueError('Unknown server: {}'.format(args.server))

    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    args.deepfakes_model_url = (
        server_url + 'v3/manipulated_sequences/Deepfakes/models/'
    )
    return args


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * max(duration, 1)))
    percent = int(count * block_size * 100 / max(total_size, 1))
    sys.stdout.write(
        '\rProgress: %d%%, %d MB, %d KB/s, %d seconds passed' %
        (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


def download_file(url, out_file, report_progress=False):
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(out_file):
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        if report_progress:
            urllib.request.urlretrieve(url, out_file_tmp, reporthook=reporthook)
        else:
            urllib.request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        tqdm.write('WARNING: skipping existing file ' + out_file)


def download_files(filenames, base_url, output_path, report_progress=True):
    os.makedirs(output_path, exist_ok=True)
    if report_progress:
        filenames = tqdm(filenames)
    for filename in filenames:
        download_file(base_url + filename, join(output_path, filename))


def main(args):
    print('By pressing any key you confirm you have agreed to the FaceForensics terms of use:')
    print(args.tos_url)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    input('')

    c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
    c_type = args.type
    c_compression = args.compression
    num_videos = args.num_videos
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    for dataset in c_datasets:
        dataset_path = DATASETS[dataset]

        # Special case: original youtube zip
        if 'original_youtube_videos' in dataset:
            print('Downloading original youtube videos (~40 GB).')
            suffix = 'info' if 'info' in dataset_path else ''
            download_file(
                args.base_url + '/' + dataset_path,
                out_file=join(output_path, 'downloaded_videos{}.zip'.format(suffix)),
                report_progress=True,
            )
            return

        print('\nDownloading {} of dataset "{}"'.format(c_type, dataset_path))

        # Get file list from server
        if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
            filepaths = json.loads(
                urllib.request.urlopen(
                    args.base_url + '/' + DEEPFAKES_DETECTION_URL
                ).read().decode('utf-8')
            )
            if 'actors' in dataset_path:
                filelist = filepaths['actors']
            else:
                filelist = filepaths['DeepFakesDetection']

        elif 'original' in dataset_path:
            file_pairs = json.loads(
                urllib.request.urlopen(
                    args.base_url + '/' + FILELIST_URL
                ).read().decode('utf-8')
            )
            filelist = []
            for pair in file_pairs:
                filelist += pair

        else:
            file_pairs = json.loads(
                urllib.request.urlopen(
                    args.base_url + '/' + FILELIST_URL
                ).read().decode('utf-8')
            )
            filelist = []
            for pair in file_pairs:
                filelist.append('_'.join(pair))
                if c_type != 'models':
                    filelist.append('_'.join(pair[::-1]))

        # Limit number of videos if requested
        if num_videos is not None and num_videos > 0:
            print('Limiting download to first {} videos'.format(num_videos))
            filelist = filelist[:num_videos]

        dataset_videos_url = args.base_url + '{}/{}/{}/'.format(
            dataset_path, c_compression, c_type
        )

        if c_type == 'videos':
            dataset_output_path = join(output_path, dataset_path, c_compression, c_type)
            print('Output path: {}'.format(dataset_output_path))
            filelist = [filename + '.mp4' for filename in filelist]
            download_files(filelist, dataset_videos_url, dataset_output_path)

        elif c_type == 'masks':
            dataset_output_path = join(output_path, dataset_path, c_type, 'videos')
            print('Output path: {}'.format(dataset_output_path))
            if 'original' in dataset:
                if args.dataset != 'all':
                    print('Only videos available for original data. Aborting.')
                    return
                else:
                    print('Only videos for original data. Skipping.')
                    continue
            if 'FaceShifter' in dataset:
                print('Masks not available for FaceShifter. Aborting.')
                return
            filelist = [filename + '.mp4' for filename in filelist]
            dataset_mask_url = args.base_url + '{}/masks/videos/'.format(dataset_path)
            download_files(filelist, dataset_mask_url, dataset_output_path)

        else:  # models
            if dataset != 'Deepfakes' and c_type == 'models':
                print('Models only available for Deepfakes. Aborting.')
                return
            dataset_output_path = join(output_path, dataset_path, c_type)
            print('Output path: {}'.format(dataset_output_path))
            for folder in tqdm(filelist):
                folder_base_url = args.deepfakes_model_url + folder + '/'
                folder_output_path = join(dataset_output_path, folder)
                download_files(
                    DEEPFAKES_MODEL_NAMES, folder_base_url,
                    folder_output_path, report_progress=False,
                )


if __name__ == '__main__':
    args = parse_args()
    main(args)