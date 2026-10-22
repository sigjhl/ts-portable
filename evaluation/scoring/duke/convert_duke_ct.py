"""Convert the 29 Duke CT series (DSDS_0001-0029, AXIAL) and their per-slice DICOM masks to NIfTI.
Image: int16 HU (slope/intercept applied), RAS affine from IPP/IOP/PixelSpacing.
Mask: uint8 {0,1} in the image grid, slices matched by position along the slice normal.

    python scoring/duke/convert_duke_ct.py Spleen_Release_Directory OUT/images OUT/masks

Spleen_Release_Directory is the unpacked Duke Spleen Data Set (Zenodo 7636640); DSDS_0001-0029 are
the CT series. Writes conversion_report.json next to OUT/images."""
import sys, os, glob, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pydicom, nibabel as nib

root, out_img, out_msk = sys.argv[1:4]
os.makedirs(out_img, exist_ok=True); os.makedirs(out_msk, exist_ok=True)
report = []

def load(d, pixels=True):
    sl = []
    for f in glob.glob(d + '/*.dcm'):
        ds = pydicom.dcmread(f)
        sl.append(ds)
    iop = np.array(sl[0].ImageOrientationPatient, float)
    n = np.cross(iop[:3], iop[3:])
    sl.sort(key=lambda s: float(np.dot(n, np.array(s.ImagePositionPatient, float))))
    return sl, iop, n

def affine_ras(sl, iop, n):
    r, c = iop[:3], iop[3:]
    ps = [float(x) for x in sl[0].PixelSpacing]   # row spacing, col spacing
    p0 = np.array(sl[0].ImagePositionPatient, float)
    if len(sl) > 1:
        dz = (np.array(sl[-1].ImagePositionPatient, float) - p0) / (len(sl) - 1)
    else:
        dz = n * float(sl[0].SliceThickness)
    A = np.eye(4)
    A[:3, 0] = r * ps[1]   # array axis 0 = column index (x along row direction)
    A[:3, 1] = c * ps[0]   # array axis 1 = row index
    A[:3, 2] = dz
    A[:3, 3] = p0
    return np.diag([-1, -1, 1, 1]) @ A          # LPS -> RAS

for i in range(1, 30):
    case = f'DSDS_{i:04d}'
    isl, iop, n = load(f'{root}/{case}/AXIAL/Images')
    msl, miop, _ = load(f'{root}/{case}/AXIAL/Masks')
    assert np.allclose(iop, miop, atol=1e-4), case
    vol = np.stack([s.pixel_array.astype(np.float64) * float(getattr(s, 'RescaleSlope', 1)) + float(getattr(s, 'RescaleIntercept', 0)) for s in isl], axis=-1)
    vol = np.transpose(vol, (1, 0, 2))           # (rows, cols, z) -> (x=cols, y=rows, z)
    m = np.stack([(s.pixel_array > 0).astype(np.uint8) for s in msl], axis=-1)
    m = np.transpose(m, (1, 0, 2))
    zi = np.array([np.dot(n, np.array(s.ImagePositionPatient, float)) for s in isl])
    zm = np.array([np.dot(n, np.array(s.ImagePositionPatient, float)) for s in msl])
    note = 'positions match'
    if len(zi) != len(zm):
        raise SystemExit(f'{case}: slice count differs')
    if not np.allclose(zi, zm, atol=0.05):
        # Same count and spacing but shifted: pick the order (as-is or reversed) that puts the mask on soft tissue.
        cand = {}
        for name, mm in (('shifted, same order', m), ('shifted, reversed order', m[:, :, ::-1])):
            hu = vol[mm > 0]
            cand[name] = (float(np.median(hu)), float(np.mean((hu > 0) & (hu < 250))))
        best = max(cand, key=lambda k: cand[k][1])
        if best.endswith('reversed order'): m = m[:, :, ::-1]
        note = f'mask z offset {zm[0]-zi[0]:+.1f} mm; aligned by slice index ({best}); candidates median HU / frac 0-250: {cand}'
    A = affine_ras(isl, iop, n)
    img = nib.Nifti1Image(np.round(vol).astype(np.int16), A); img.set_qform(A, 1); img.set_sform(A, 1)
    msk = nib.Nifti1Image(np.ascontiguousarray(m), A); msk.set_qform(A, 1); msk.set_sform(A, 1)
    nib.save(img, f'{out_img}/{case}.nii.gz'); nib.save(msk, f'{out_msk}/{case}_spleen.nii.gz')
    vox = abs(np.linalg.det(A[:3, :3]))
    hu = vol[m > 0]
    rec = dict(case=case, shape=list(vol.shape), spacing=[round(float(x), 4) for x in nib.affines.voxel_sizes(A)],
               transfer_syntax=str(isl[0].file_meta.TransferSyntaxUID), manufacturer=str(getattr(isl[0], 'Manufacturer', '')),
               slice_thickness=str(getattr(isl[0], 'SliceThickness', '')), contrast=str(getattr(isl[0], 'ContrastBolusAgent', '')),
               mask_ml=round(float(m.sum()) * vox / 1000, 1), mask_median_hu=round(float(np.median(hu)), 1), note=note)
    report.append(rec); print(json.dumps(rec))
json.dump(report, open(os.path.join(os.path.dirname(out_img.rstrip('/')), 'conversion_report.json'), 'w'), indent=1)
