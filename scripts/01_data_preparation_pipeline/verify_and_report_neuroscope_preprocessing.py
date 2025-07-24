import os
import json
import logging
import SimpleITK as sitk
import numpy as np
import time
from typing import Dict, Any


def configure_logging() -> None:
    """
    Configure logging for verification.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def verify_subject(
    section: str,
    subj_id: str,
    subj_dir: str,
    target_spacing: tuple
) -> Dict[str, Any]:
    """
    Verify preprocessing for one subject.
    Returns metrics and flags for failures.
    """
    metrics: Dict[str, Any] = {"section": section, "subject": subj_id}
    files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    if not files:
        logging.warning("No NIfTI files found for subject %s/%s", section, subj_id)
        metrics['spacing_ok'] = False
        metrics['intensity_ok'] = False
        metrics['mask_overlap_ok'] = False
        metrics['all_ok'] = False
        return metrics

    imgs = {f: sitk.ReadImage(os.path.join(subj_dir, f)) for f in files}

    # check spacing
    spacing_ok = True
    for name, img in imgs.items():
        if any(abs(img.GetSpacing()[i]-target_spacing[i])>1e-3 for i in range(3)):
            spacing_ok = False
    metrics['spacing_ok'] = spacing_ok

    # check intensity
    intens_ok = True
    for name,img in imgs.items():
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if arr.min() < -0.01 or arr.max() > 1.01:
            intens_ok = False
    metrics['intensity_ok'] = intens_ok

    # check mask overlap
    mask = sitk.OtsuThreshold(imgs[files[0]],0,1,200)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    overlap_ok = True
    for name,img in imgs.items():
        arr = sitk.GetArrayFromImage(img)>0
        overlap = (arr & mask_arr).sum()/max(mask_arr.sum(),1)
        if overlap<0.9:
            overlap_ok = False
    metrics['mask_overlap_ok'] = overlap_ok

    metrics['all_ok'] = spacing_ok and intens_ok and overlap_ok
    return metrics


def main():
    configure_logging()
    start=time.time()

    base = os.path.expanduser('~/Downloads/neuroscope')
    preproc = os.path.join(base,'preprocessed')
    meta_file = os.path.join(base,'scripts','neuroscope_dataset_metadata_splits.json')
    report_file = os.path.join(base,'scripts','preprocessing_verification_report_v2.json')
    target_spacing=(1.0,1.0,1.0)

    # load metadata
    with open(meta_file,'r') as f:
        meta=json.load(f)

    results=[]
    for section in ('brats','upenn'):
        for sid,info in meta.get(section,{}).get('valid_subjects',{}).items():
            if info.get('split')!='train': continue
            subj_dir=os.path.join(preproc,section,sid)
            if not os.path.isdir(subj_dir):
                logging.error("Missing preproc dir %s/%s",section,sid)
                results.append({"section":section,"subject":sid,"all_ok":False})
                continue
            metrics=verify_subject(section,sid,subj_dir,target_spacing)
            results.append(metrics)

    # write full report
    with open(report_file,'w') as f:
        json.dump({"timestamp":time.strftime('%Y-%m-%d %H:%M:%S'),"results":results},f,indent=2)
    logging.info("Wrote detailed report to %s",report_file)

    # summary
    total=len(results)
    ok_count=sum(1 for r in results if r['all_ok'])
    spacing_fail=sum(1 for r in results if not r['spacing_ok'])
    intens_fail=sum(1 for r in results if not r['intensity_ok'])
    overlap_fail=sum(1 for r in results if not r['mask_overlap_ok'])

    print(f"\nPreprocessing Verification Summary:")
    print(f"  Total train subjects: {total}")
    print(f"  Fully passed:        {ok_count} ({ok_count/total:.1%})")
    print(f"  Spacing failures:    {spacing_fail}")
    print(f"  Intensity failures:  {intens_fail}")
    print(f"  Mask overlap fails:  {overlap_fail}\n")

    logging.info("Summary generated in %.2f seconds",time.time()-start)

if __name__=='__main__':
    main()
