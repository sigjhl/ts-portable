"""DSDS_0021: search for the slice offset that puts the reference mask on soft tissue (Supplement,
"Duke Spleen Data Set"). The series is excluded from all analyses.

    python scoring/duke/check_dsds0021_alignment.py OUT    (OUT/images, OUT/masks from convert_duke_ct.py)
"""
import sys, numpy as np, nibabel as nib
d=sys.argv[1]; c='DSDS_0021'
v=np.asarray(nib.load(f'{d}/images/{c}.nii.gz').dataobj); m0=np.asarray(nib.load(f'{d}/masks/{c}_spleen.nii.gz').dataobj)>0
nz=v.shape[2]; zs=np.where(m0.any((0,1)))[0]; print('mask slices', zs.min(), zs.max(), 'of', nz, 'vox', m0.sum())
res=[]
for name,m in (('same',m0),('rev',m0[:,:,::-1]),('flipx',m0[::-1]),('flipy',m0[:,::-1])):
    for s in range(-nz+1,nz):
        mm=np.zeros_like(m)
        if s>=0: mm[:,:,s:]=m[:,:,:nz-s]
        else: mm[:,:,:nz+s]=m[:,:,-s:]
        if mm.sum()<0.95*m.sum(): continue
        h=v[mm]; res.append((np.mean((h>0)&(h<250)), name, s, float(np.median(h))))
res.sort(reverse=True)
for r in res[:8]: print('frac=%.3f %s shift=%d median=%.0f'%r)
