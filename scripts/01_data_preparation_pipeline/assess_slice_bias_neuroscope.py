import os, json, logging
import SimpleITK as sitk
import numpy as np

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_metadata(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path,'r') as f:
        return json.load(f)

def generate_brain_mask(image):
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    return sitk.BinaryFillhole(mask)

def compute_slice_variation(image, mask):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    m = sitk.GetArrayFromImage(mask).astype(bool)
    # arr[z,y,x], so axis=1,2 for mask sum; iterate z
    slice_means = []
    for z in range(arr.shape[0]):
        slice_vals = arr[z][m[z]]
        if slice_vals.size>0:
            slice_means.append(slice_vals.mean())
    slice_means = np.array(slice_means)
    if slice_means.size<2:
        return np.nan
    global_std = arr[m].std()
    ratio = slice_means.std() / (global_std + 1e-8)
    return float(ratio)

def assess_dataset(metadata, section, modalities):
    results = {}
    for subj, info in metadata[section]['valid_subjects'].items():
        subj_res = {}
        for suffix in modalities:
            path = info.get(suffix)
            if not path or not os.path.isfile(path):
                continue
            img = sitk.ReadImage(path)
            mask = generate_brain_mask(img)
            ratio = compute_slice_variation(img, mask)
            subj_res[suffix] = ratio
            logging.info('%s %s slice_ratio=%.3f', section, subj, ratio)
        results[subj] = subj_res
    return results

def main():
    configure_logging()
    base = '/Volumes/USB Drive/neuroscope'
    meta = load_metadata(os.path.join(base,'scripts','neuroscope_dataset_metadata_splits.json'))
    brats_mods = ['_t1.nii.gz','_t1Gd.nii.gz','_t2.nii.gz','_flair.nii.gz']
    upenn_mods = ['_T1.nii.gz','_T1GD.nii.gz','_T2.nii.gz','_FLAIR.nii.gz']

    logging.info('Assessing slice bias for BraTS')
    brats = assess_dataset(meta, 'brats', brats_mods)
    logging.info('Assessing slice bias for UPenn')
    upenn = assess_dataset(meta, 'upenn', upenn_mods)

    out = {'brats': brats, 'upenn': upenn}
    out_path = os.path.join(base,'scripts','neuroscope_slice_bias_assessment.json')
    with open(out_path,'w') as f:
        json.dump(out, f, indent=2)
    logging.info('Slice bias assessment saved to %s', out_path)

if __name__=='__main__':
    main()