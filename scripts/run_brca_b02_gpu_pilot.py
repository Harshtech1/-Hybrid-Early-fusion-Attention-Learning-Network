#!/usr/bin/env python3
"""Execution-locked, one-patient B02 GPU feature pilot."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXECUTION_AUTHORIZED=False
EXECUTION_AUTH_SHA256="PENDING_SEPARATE_EXACT_B02_GPU_AUTHORIZATION"
AUTH=ROOT/'multiscale_feature_pilot/config/brca_b02_gpu_execution_authorization.yaml'
PATIENT='TCGA-BH-A0BG'; SLIDE='TCGA-BH-A0BG-01Z-00-DX1.0838FB7F-8C85-4687-9F70-D136A1063383.svs'; UUID='c5331e5e-10b4-4979-958b-d4592a2805de'
WSI=Path('/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B02.incoming')/UUID/SLIDE
COORD=Path('/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B02.coordinates')
OUTPUT=Path('/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B02.features')
OMIC=Path('/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip')
CHECKPOINT=Path('/home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')
OFFICIAL=Path('/teamspace/studios/this_studio/healnet')
WSI_SIZE=724114911; WSI_MD5='a8c6e730df401ff67e1a1e52a6cb6307'; WSI_SHA='df85b3c048b18ae0a5b9414e7e220110d98891f73f28189849e6e602d1743741'
COORD_SHA='2b3e5dd754ebb4ca4ec26f3e017e21548b0115dc2d0517ae83146d8f7ec52ba2'; POLICY_SHA='8f903489da7a653665fd7be8aced250c7656350fa0bd6370b168eb3e5baa0953'
CHECKPOINT_SHA='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'; OFFICIAL_HEAD='28ba5da6ab99fd8069972c22e986d83edb658dd4'
DIMS=((89291,72971),(22322,18242),(5580,4560),(2790,2280)); DOWNSAMPLES=(1.0,4.0001494261056205,16.002191803433313,32.004383606866625)
COUNTS=(7158,1862); TOTAL=9020
BOUND=(Path('scripts/run_brca_b02_gpu_pilot.py'),Path('multiscale_feature_pilot/config/brca_b02_gpu_preexecution.yaml'),Path('multiscale_feature_pilot/provenance/brca_b02_coordinate_execution_result.yaml'),Path('multiscale_feature_pilot/config/brca_b02_scale_coordinate_policy.yaml'),Path('multiscale_feature_pilot/src/brca_compact_feature_artifacts.py'),Path('multiscale_feature_pilot/src/brca_coordinate_artifacts.py'),Path('multiscale_feature_pilot/src/brca_omic.py'),Path('multiscale_feature_pilot/src/feature_extraction.py'),Path('multiscale_feature_pilot/src/provenance.py'),Path('multiscale_feature_pilot/src/supervisor_healnet_smoke.py'))
ALLOWED={'M reports/blca_one_patient_multiscale_pilot.md',' M reports/brca_compact_artifact_and_recovery_design.md','?? reports/brca_supervisor_progress_report.html'}
class ExecutionLocked(RuntimeError): pass
def _require_authorized():
    if not EXECUTION_AUTHORIZED: raise ExecutionLocked('B02 GPU execution is locked pending separate exact authorization')
    if len(EXECUTION_AUTH_SHA256)!=64 or any(c not in '0123456789abcdef' for c in EXECUTION_AUTH_SHA256): raise ExecutionLocked('B02 execution authorization SHA256 is not pinned')
def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def git(*args,cwd=ROOT): return subprocess.run(('git',*args),cwd=cwd,check=True,text=True,capture_output=True).stdout.strip()
def _execute(commit):
    import h5py, numpy as np, openslide, torch, torchvision, yaml
    sys.path.insert(0,str(ROOT))
    from multiscale_feature_pilot.src.brca_compact_feature_artifacts import CompactFeatureMetadata,publish_compact_feature_artifacts,validate_compact_feature_artifacts
    from multiscale_feature_pilot.src.brca_coordinate_artifacts import validate_brca_coordinate_artifacts
    from multiscale_feature_pilot.src.brca_omic import BRCA_RELEASE_ARCHIVE_SHA256,load_official_brca_patient_omics
    from multiscale_feature_pilot.src.feature_extraction import PatchBranchSpec,StreamingOpenSlideDataset,build_resnet50_imagenet1k_v2,extract_feature_matrix
    from multiscale_feature_pilot.src.provenance import BranchProvenanceSpec,build_two_scale_provenance
    from multiscale_feature_pilot.src.supervisor_healnet_smoke import run_one_patient_supervisor_healnet_smoke
    if git('rev-parse','HEAD')!=commit: raise RuntimeError('HEAD drift')
    status=set(filter(None,git('status','--short').splitlines()))
    if status-ALLOWED: raise RuntimeError(f'unexpected Git status {status-ALLOWED}')
    for rel in BOUND:
        committed=subprocess.run(('git','show',f'HEAD:{rel.as_posix()}'),cwd=ROOT,check=True,stdout=subprocess.PIPE).stdout
        if (ROOT/rel).read_bytes()!=committed: raise RuntimeError(f'bound source drift {rel}')
    if digest(AUTH)!=EXECUTION_AUTH_SHA256: raise RuntimeError('authorization drift')
    auth=yaml.safe_load(AUTH.read_text())
    if auth.get('status')!='B02_GPU_FEATURE_PILOT_AUTHORIZED' or auth['scope']['combined_shape']!=[9020,2048]: raise RuntimeError('authorization semantics')
    if OUTPUT.exists() or OUTPUT.is_symlink() or list(OUTPUT.parent.glob('.BRCA_BATCH_B02.features.staging.*')): raise RuntimeError('output collision')
    if git('rev-parse','HEAD',cwd=OFFICIAL)!=OFFICIAL_HEAD or git('status','--porcelain',cwd=OFFICIAL): raise RuntimeError('official HEALNet drift')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8': raise RuntimeError('CUBLAS_WORKSPACE_CONFIG required before CUDA')
    if WSI.is_symlink() or not WSI.is_file() or WSI.stat().st_size!=WSI_SIZE or digest(WSI,'md5')!=WSI_MD5 or digest(WSI)!=WSI_SHA: raise RuntimeError('WSI identity drift')
    record=validate_brca_coordinate_artifacts(COORD,expected_manifest_sha256=COORD_SHA)
    coords=[]
    for name,count in zip(('scale_2x','scale_4x'),COUNTS):
        if record.branch_for(name).coordinate_count!=count: raise RuntimeError('coordinate count drift')
        with h5py.File(COORD/f'{name}_coordinates.h5','r') as h: coords.append(torch.from_numpy(np.asarray(h['coords'],dtype=np.int64)).contiguous())
    omic=load_official_brca_patient_omics(OMIC,case_id=PATIENT,slide_id=SLIDE)
    if omic.source_row_index!='472' or digest(OMIC)!=BRCA_RELEASE_ARCHIVE_SHA256: raise RuntimeError('Omic drift')
    if CHECKPOINT.stat().st_size!=102540417 or digest(CHECKPOINT)!=CHECKPOINT_SHA: raise RuntimeError('checkpoint drift')
    with openslide.OpenSlide(str(WSI)) as slide:
        if tuple(slide.level_dimensions)!=DIMS or any(abs(a-b)>1e-10 for a,b in zip(slide.level_downsamples,DOWNSAMPLES)): raise RuntimeError('header drift')
    torch.manual_seed(0); torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('exactly one CUDA GPU required; CPU fallback forbidden')
    if 'Tesla T4' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0)!=(7,5): raise RuntimeError('Tesla T4 capability 7.5 required')
    device=torch.device('cuda:0'); started=datetime.now(timezone.utc); start=time.perf_counter()
    synthetic=run_one_patient_supervisor_healnet_smoke(official_repo=OFFICIAL,wsi=torch.zeros((1,TOTAL,2048),device=device),rna=torch.zeros((1,1,1558),device=device),mutation=torch.zeros((1,1,21),device=device),cnv=torch.zeros((1,1,1333),device=device))
    model=build_resnet50_imagenet1k_v2(CHECKPOINT).to(device).eval()
    r2=extract_feature_matrix(StreamingOpenSlideDataset(WSI,PatchBranchSpec('scale_2x',coords[0],0,256)),model,device=device,batch_size=32,num_workers=2)
    r4=extract_feature_matrix(StreamingOpenSlideDataset(WSI,PatchBranchSpec('scale_4x',coords[1],1,256)),model,device=device,batch_size=32,num_workers=2)
    combined=torch.cat((r2.features,r4.features),dim=0).contiguous()
    if combined.shape==(9020,2048):
        pass
    else:
        raise RuntimeError('combined shape contract')
    if combined.dtype!=torch.float32 or not torch.isfinite(combined).all(): raise RuntimeError('combined numerical contract')
    real=run_one_patient_supervisor_healnet_smoke(official_repo=OFFICIAL,wsi=combined.unsqueeze(0).to(device),rna=omic.rna.to(device),mutation=omic.mutation.to(device),cnv=omic.cnv.to(device))
    rows=build_two_scale_provenance(scale_2x=BranchProvenanceSpec('scale_2x',coords[0],0,.501,.501),scale_4x=BranchProvenanceSpec('scale_4x',coords[1],1,1.002037431239458,1.002037431239458),scale_2x_count=7158,scale_4x_count=1862)
    metadata=CompactFeatureMetadata(PATIENT,SLIDE,UUID,WSI_SHA,COORD_SHA,BRCA_RELEASE_ARCHIVE_SHA256,CHECKPOINT_SHA,POLICY_SHA,commit,7158,1862)
    manifest=publish_compact_feature_artifacts(OUTPUT,combined_features=combined,row_provenance=rows,metadata=metadata,preserve_failed_staging=True)
    anchor=digest(OUTPUT/'compact_manifest.json'); validate_compact_feature_artifacts(OUTPUT,expected_manifest_sha256=anchor)
    return {'status':'BRCA_BATCH_B02_GPU_FEATURE_PILOT_SUCCESS','started_at_utc':started.isoformat(),'finished_at_utc':datetime.now(timezone.utc).isoformat(),'total_seconds':time.perf_counter()-start,'source_commit':commit,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'torchvision':torchvision.__version__,'counts':{'scale_2x':7158,'scale_4x':1862,'total':TOTAL},'feature_shapes':{'scale_2x':list(r2.features.shape),'scale_4x':list(r4.features.shape),'combined':list(combined.shape),'natural':[1,TOTAL,2048]},'timings':{'scale_2x':r2.streaming_extraction_seconds,'scale_4x':r4.streaming_extraction_seconds},'peak_gpu_memory_bytes':max(r2.peak_gpu_memory_bytes,r4.peak_gpu_memory_bytes),'synthetic_healnet':asdict(synthetic),'real_healnet':asdict(real),'artifact':{'directory':str(OUTPUT),'manifest_sha256':anchor,'manifest':manifest,'total_file_bytes':sum(p.stat().st_size for p in OUTPUT.iterdir())},'operations':{'training':0,'backward':0,'optimizer_steps':0,'amp':0,'tf32':0,'coordinate_generation':0,'deletions':0,'other_patients':0}}
def run(commit):
    _require_authorized()
    return _execute(commit)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--expected-source-commit',required=True); a=p.parse_args()
    try: result=run(a.expected_source_commit)
    except Exception as e: print(json.dumps({'status':'BLOCKED','error':f'{type(e).__name__}: {e}'},sort_keys=True)); return 1
    print(json.dumps(result,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
