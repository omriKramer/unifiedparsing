import itertools
from pathlib import Path

import PIL
import numpy as np
import pandas as pd

from broden_dataset_utils.joint_dataset import BrodenDataset


def main(root):
    compare_dir = root / 'reindexed2'
    broden = BrodenDataset(root)
    classes = np.arange(broden.nr['object'])
    class_pix = np.zeros_like(classes)
    total = 0
    for record in broden.record_list['validation']:
        if record['dataset'] not in ['pascal', 'ade20k']:
            continue

        total += 1
        data = broden.resolve_record(record)
        ds = broden.data_sets[record['dataset']]
        filename = Path(ds.filename(record['file_index']))

        seg_obj = data['seg_obj']
        obj2 = PIL.Image.open(compare_dir / f'{filename.stem}_obj.png').convert('I')
        obj2 = np.asarray(obj2)
        assert obj2.shape == seg_obj.shape
        assert np.all(obj2 == seg_obj)
        is_class = seg_obj == classes[:, None, None]
        class_pix += np.sum(is_class, axis=(1, 2))

    print(f'resolved {total} records')
    print(np.any(class_pix < 10))
    for c, n_pix in enumerate(class_pix[1:], start=1):
        if n_pix < 10:
            print(broden.names['object'][c], n_pix)
    print(broden.names['object'][int(class_pix.argmin())], np.min(class_pix))
    print(class_pix)
    np.save('obj_val_pixel_count.npy', class_pix)


def check_part_pix_count(root, validation):
    broden = BrodenDataset(root)
    record_list = get_records(broden, validation)

    n_pix = {o: np.zeros(len(broden.object_part[o]), dtype=int) for o in broden.object_with_part}
    for record in record_list:
        if record['dataset'] not in ['pascal', 'ade20k']:
            continue

        data = broden.resolve_record(record)
        seg_part = data['batch_seg_part']
        assert len(broden.object_with_part) == len(seg_part) == len(data['valid_part'])

        for o, o_part_seg, v in zip(broden.object_with_part, seg_part, data['valid_part']):
            if not v:
                continue
            assert o_part_seg.ndim == 2
            n_parts = len(broden.object_part[o])
            n_pix[o] += np.histogram(o_part_seg, bins=n_parts, range=(0, n_parts))[0]

    for o, count in n_pix.items():
        for i, c in enumerate(count):
            if c < 10:
                part = broden.object_part[o][i]
                print(broden.names['object'][o], broden.names['part'][part], c)


def get_files(root, validation):
    broden = BrodenDataset(root)
    record_list = get_records(broden, validation)

    items = []
    for record in record_list:
        if record['dataset'] not in ['pascal', 'ade20k']:
            continue

        ds = broden.data_sets[record['dataset']]
        filename = Path(ds.filename(record['file_index']))
        items.append(filename)

    df = pd.DataFrame(items)
    fn = f'is_{"val" if validation else "train"}.csv'
    df.to_csv(fn, index=False)


def get_records(broden, validation):
    if validation:
        return broden.record_list['validation']

    return broden.record_list['train'][0]


if __name__ == '__main__':
    p = Path('/Volumes/waic/omrik/unifiedparsing/broden_dataset')
    # main(p)
    # get_files(p, False)
    check_part_pix_count(p, True)
