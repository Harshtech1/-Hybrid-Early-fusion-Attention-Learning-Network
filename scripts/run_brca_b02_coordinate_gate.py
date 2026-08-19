#!/usr/bin/env python3
"""Execute the exact authorized B02 single-mask coordinate gate."""
from __future__ import annotations
import argparse, ctypes, errno, hashlib, json, os, stat, subprocess, sys, uuid
from pathlib import Path
import h5py, numpy as np, openslide, yaml

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from multiscale_feature_pilot.src.brca_coordinate_artifacts import BRANCH_FILENAMES,CoordinateBranchMetadata,MANIFEST_FILENAME,MANIFEST_SHA256_FILENAME,SCHEMA,sha256_file,validate_brca_coordinate_artifacts
from multiscale_feature_pilot.src.brca_omic import BRCA_RELEASE_ARCHIVE_SHA256,load_brca_patient_omics
from multiscale_feature_pilot.src.brca_q25_coordinates import generate_level_0_lattice_coordinates,segment_tissue_contours

WORKSPACE=ROOT.parent; DATA=WORKSPACE/'brca_pilot_data'
AUTH=ROOT/'multiscale_feature_pilot/config/brca_b02_coordinate_execution_authorization.yaml'
POLICY=ROOT/'multiscale_feature_pilot/config/brca_b02_scale_coordinate_policy.yaml'
WSI=DATA/'BRCA_BATCH_B02.incoming/c5331e5e-10b4-4979-958b-d4592a2805de/TCGA-BH-A0BG-01Z-00-DX1.0838FB7F-8C85-4687-9F70-D136A1063383.svs'
OMIC=WORKSPACE/'Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip'
DEST=DATA/'BRCA_BATCH_B02.coordinates'
PATIENT='TCGA-BH-A0BG'; SLIDE='TCGA-BH-A0BG-01Z-00-DX1.0838FB7F-8C85-4687-9F70-D136A1063383.svs'; UUID='c5331e5e-10b4-4979-958b-d4592a2805de'
SIZE=724114911; MD5='a8c6e730df401ff67e1a1e52a6cb6307'; SHA='df85b3c048b18ae0a5b9414e7e220110d98891f73f28189849e6e602d1743741'
AUTH_SHA='377d032f267dda049933e14569a53d0ae8185d94dde6e6d023cee309e16e43fd'; POLICY_SHA='8f903489da7a653665fd7be8aced250c7656350fa0bd6370b168eb3e5baa0953'
DIMS=((89291,72971),(22322,18242),(5580,4560),(2790,2280)); DS=(1.0,4.0001494261056205,16.002191803433313,32.004383606866625); MPP=(.2505,.2505); MASK=(5580,4560)
CRITICAL=(Path('scripts/run_brca_b02_coordinate_gate.py'),Path('multiscale_feature_pilot/config/brca_b02_coordinate_execution_authorization.yaml'),Path('multiscale_feature_pilot/config/brca_b02_scale_coordinate_policy.yaml'),Path('multiscale_feature_pilot/src/brca_q25_coordinates.py'),Path('multiscale_feature_pilot/src/brca_coordinate_artifacts.py'),Path('multiscale_feature_pilot/src/brca_omic.py'))
ALLOWED={'M reports/blca_one_patient_multiscale_pilot.md',' M reports/brca_compact_artifact_and_recovery_design.md','?? reports/brca_supervisor_progress_report.html'}
class GateError(RuntimeError): pass
def req(x,m):
    if not x: raise GateError(m)
def digest(p,a='sha256'):
    h=hashlib.new(a)
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
    return h.hexdigest()
def git(*a): return subprocess.run(('git',*a),cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()
def preflight(commit):
    req(len(commit)==40 and git('rev-parse','HEAD')==commit,'HEAD drift')
    statuses=set(filter(None,git('status','--short').splitlines())); req(statuses<=ALLOWED,f'unexpected Git status {statuses-ALLOWED}')
    for rel in CRITICAL:
        committed=subprocess.run(('git','show',f'HEAD:{rel.as_posix()}'),cwd=ROOT,check=True,stdout=subprocess.PIPE).stdout
        req((ROOT/rel).read_bytes()==committed,f'critical drift {rel}')
    req(digest(AUTH)==AUTH_SHA and digest(POLICY)==POLICY_SHA,'authorization or policy drift')
    auth=yaml.safe_load(AUTH.read_text()); req(auth['status']=='AUTHORIZED_B02_SINGLE_MASK_READ_AND_COORDINATE_PUBLICATION','authorization status')
    req(not os.path.lexists(DEST) and not tuple(DATA.glob('.BRCA_BATCH_B02.coordinates.staging.*')),'output collision')
    req(not WSI.is_symlink() and WSI.is_file() and WSI.stat().st_size==SIZE,'WSI identity')
    req(digest(WSI,'md5')==MD5 and digest(WSI)==SHA,'WSI hashes')
    o=load_brca_patient_omics(OMIC,case_id=PATIENT,slide_id=SLIDE,expected_archive_sha256=BRCA_RELEASE_ARCHIVE_SHA256)
    req(o.source_row_index=='472' and all(t.device.type=='cpu' and bool(t.isfinite().all()) for t in (o.rna,o.mutation,o.cnv)),'Omic drift')
def read_mask():
    fd=os.open(WSI,os.O_RDONLY|getattr(os,'O_CLOEXEC',0)|getattr(os,'O_NOFOLLOW',0)); calls=0
    try:
        before=os.fstat(fd); req(stat.S_ISREG(before.st_mode) and before.st_size==SIZE,'held WSI drift')
        slide=openslide.OpenSlide(f'/proc/self/fd/{fd}')
        try:
            dims=tuple(tuple(map(int,x)) for x in slide.level_dimensions); ds=tuple(float(x) for x in slide.level_downsamples)
            mpp=(float(slide.properties[openslide.PROPERTY_NAME_MPP_X]),float(slide.properties[openslide.PROPERTY_NAME_MPP_Y]))
            req(dims==DIMS and ds==DS and mpp==MPP,'header drift')
            image=slide.read_region((0,0),2,MASK); calls+=1
            mask=np.ascontiguousarray(np.asarray(image,dtype=np.uint8)); req(mask.shape==(4560,5580,4),'mask shape')
        finally: slide.close()
        req(calls==1 and os.fstat(fd).st_ino==before.st_ino,'read count or descriptor drift')
    finally: os.close(fd)
    req(digest(WSI,'md5')==MD5 and digest(WSI)==SHA,'final WSI drift')
    return mask,hashlib.sha256(mask.tobytes(order='C')).hexdigest()
def meta(branch,mask_sha,contours,holes):
    is2=branch=='scale_2x'; level=0 if is2 else 1
    return CoordinateBranchMetadata(branch=branch,patient_id=PATIENT,slide_id=SLIDE,gdc_file_uuid=UUID,wsi_filename=SLIDE,wsi_size_bytes=SIZE,wsi_md5=MD5,wsi_sha256=SHA,level_0_dimensions=DIMS[0],source_level=level,source_level_dimensions=DIMS[level],openslide_reported_source_downsample=DS[level],source_patch_size=(512,512) if is2 else (256,256),output_patch_size=(256,256),level_0_declared_footprint=(512,512) if is2 else (1024,1024),level_0_step=(512,512) if is2 else (1024,1024),target_mpp=.5 if is2 else 1.0,effective_mpp=(.501,.501) if is2 else (1.002037431239458,1.002037431239458),interpolation='PIL.Image.Resampling.LANCZOS' if is2 else 'none',resampling='linear_factor_2' if is2 else 'none',mask_level=2,mask_level_dimensions=MASK,openslide_reported_mask_downsample=DS[2],mask_image_channels=4,mask_image_sha256=mask_sha,mask_parameters={'sthresh':8,'mthresh':7,'close':4,'use_otsu':False,'a_t':100,'a_h':16,'max_n_holes':8,'reference_patch_size':512},contour_count=contours,retained_hole_count=holes,clam_commit='26e0b6c4873e112f1ccd74cd834894c4ab7a2934',policy_sha256=POLICY_SHA,geometry_compatibility='GLOBAL_LEVEL0_NATIVE' if is2 else 'CLAM_INT_CAST_GEOMETRY_COMPATIBLE')
def publish(c2,c4,m2,m4):
    stage=DATA/f'.BRCA_BATCH_B02.coordinates.staging.{uuid.uuid4().hex}'; req(not os.path.lexists(stage) and not os.path.lexists(DEST),'publication collision'); stage.mkdir(mode=0o700)
    branches={}
    for branch,coords,m in (('scale_2x',c2,m2),('scale_4x',c4,m4)):
        p=stage/BRANCH_FILENAMES[branch]
        with h5py.File(p,'x') as h:
            ds=h.create_dataset('coords',data=np.ascontiguousarray(coords,dtype=np.int64))
            for k,v in m.to_attributes().items(): ds.attrs[k]=v
            h.flush()
        branches[branch]={'filename':p.name,'size_bytes':p.stat().st_size,'sha256':sha256_file(p),'coordinates_sha256':hashlib.sha256(np.ascontiguousarray(coords,dtype='<i8').tobytes()).hexdigest(),'coordinate_count':int(coords.shape[0]),'attributes':m.to_attributes()}
    payload=(json.dumps({'schema':SCHEMA,'branches':branches},sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n').encode(); (stage/MANIFEST_FILENAME).write_bytes(payload)
    anchor=hashlib.sha256(payload).hexdigest(); (stage/MANIFEST_SHA256_FILENAME).write_text(f'{anchor}  {MANIFEST_FILENAME}\n',encoding='ascii',newline='')
    validate_brca_coordinate_artifacts(stage,expected_manifest_sha256=anchor)
    libc=ctypes.CDLL(None,use_errno=True); result=libc.renameat2(-100,os.fsencode(stage),-100,os.fsencode(DEST),1)
    if result!=0:
        n=ctypes.get_errno()
        if n==errno.EEXIST: raise GateError('destination appeared')
        raise OSError(n,os.strerror(n))
    return validate_brca_coordinate_artifacts(DEST,expected_manifest_sha256=anchor)
def run(commit):
    preflight(commit); mask,mask_sha=read_mask(); geom=segment_tissue_contours(mask,level_0_dimensions=DIMS[0],mask_dimensions=MASK)
    c2=generate_level_0_lattice_coordinates(level_0_dimensions=DIMS[0],level_0_patch_size=512,level_0_step=512,geometry=geom); c4=generate_level_0_lattice_coordinates(level_0_dimensions=DIMS[0],level_0_patch_size=1024,level_0_step=1024,geometry=geom)
    req(c2.shape[0]>0 and c4.shape[0]>0,'empty coordinates'); nc=len(geom.contours); nh=sum(len(x) for x in geom.holes)
    record=publish(c2,c4,meta('scale_2x',mask_sha,nc,nh),meta('scale_4x',mask_sha,nc,nh))
    return {'status':'BRCA_B02_COORDINATES_VERIFIED','manifest_sha256':record.manifest_sha256,'scale_2x_count':int(c2.shape[0]),'scale_4x_count':int(c4.shape[0]),'contour_count':nc,'retained_hole_count':nh,'read_region_calls':1,'mask_sha256':mask_sha}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--expected-source-commit',required=True); print(json.dumps(run(p.parse_args().expected_source_commit),sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
