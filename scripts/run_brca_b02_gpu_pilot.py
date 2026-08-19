#!/usr/bin/env python3
"""Fail-closed one-shot B02 compact GPU feature pilot."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import h5py,numpy as np,openslide,torch,torchvision,yaml

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from multiscale_feature_pilot.src.brca_compact_feature_artifacts import CompactFeatureMetadata,publish_compact_feature_artifacts,validate_compact_feature_artifacts
from multiscale_feature_pilot.src.brca_coordinate_artifacts import validate_brca_coordinate_artifacts
from multiscale_feature_pilot.src.brca_omic import BRCA_RELEASE_ARCHIVE_SHA256,load_official_brca_patient_omics
from multiscale_feature_pilot.src.feature_extraction import PatchBranchSpec,StreamingOpenSlideDataset,build_resnet50_imagenet1k_v2,extract_feature_matrix
from multiscale_feature_pilot.src.provenance import BranchProvenanceSpec,build_two_scale_provenance
from multiscale_feature_pilot.src.supervisor_healnet_smoke import run_one_patient_supervisor_healnet_smoke

PATIENT='TCGA-BH-A0BG'; SLIDE='TCGA-BH-A0BG-01Z-00-DX1.0838FB7F-8C85-4687-9F70-D136A1063383.svs'; UUID='c5331e5e-10b4-4979-958b-d4592a2805de'
WSI=Path('/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B02.incoming')/UUID/SLIDE
WSI_SIZE=724114911; WSI_MD5='a8c6e730df401ff67e1a1e52a6cb6307'; WSI_SHA='df85b3c048b18ae0a5b9414e7e220110d98891f73f28189849e6e602d1743741'
COORD=Path('/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B02.coordinates'); COORD_SHA='2b3e5dd754ebb4ca4ec26f3e017e21548b0115dc2d0517ae83146d8f7ec52ba2'
OUTPUT=Path('/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B02.features')
OMIC=Path('/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip')
CHECKPOINT=Path('/home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'); CHECKPOINT_SHA='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
AUTH=ROOT/'multiscale_feature_pilot/config/brca_b02_gpu_execution_authorization.yaml'; AUTH_SHA='bd2ca8270089945a9060b802c1e4c8d32da73bd24f3124cd1ceed703b1d29471'
POLICY_SHA='8f903489da7a653665fd7be8aced250c7656350fa0bd6370b168eb3e5baa0953'; OFFICIAL=Path('/teamspace/studios/this_studio/healnet'); OFFICIAL_HEAD='28ba5da6ab99fd8069972c22e986d83edb658dd4'
DIMS=((89291,72971),(22322,18242),(5580,4560),(2790,2280)); DS=(1.0,4.0001494261056205,16.002191803433313,32.004383606866625)
ALLOWED={'M reports/blca_one_patient_multiscale_pilot.md',' M reports/brca_compact_artifact_and_recovery_design.md','?? reports/brca_supervisor_progress_report.html'}
CRITICAL=(Path('scripts/run_brca_b02_gpu_pilot.py'),Path('multiscale_feature_pilot/config/brca_b02_gpu_execution_authorization.yaml'),Path('multiscale_feature_pilot/src/brca_compact_feature_artifacts.py'),Path('multiscale_feature_pilot/src/feature_extraction.py'),Path('multiscale_feature_pilot/src/provenance.py'),Path('multiscale_feature_pilot/src/supervisor_healnet_smoke.py'),Path('multiscale_feature_pilot/src/brca_omic.py'))
def digest(p,a='sha256'):
 h=hashlib.new(a)
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()
def git(*a,cwd=ROOT): return subprocess.run(('git',*a),cwd=cwd,check=True,text=True,capture_output=True).stdout.strip()
def require(x,m):
 if not x: raise RuntimeError(m)
def preflight(commit):
 require(len(commit)==40 and git('rev-parse','HEAD')==commit,'source HEAD mismatch')
 status=set(filter(None,git('status','--short').splitlines())); require(status<=ALLOWED,f'unexpected worktree changes {status-ALLOWED}')
 for rel in CRITICAL:
  committed=subprocess.run(('git','show',f'HEAD:{rel.as_posix()}'),cwd=ROOT,check=True,stdout=subprocess.PIPE).stdout
  require((ROOT/rel).read_bytes()==committed,f'critical source drift {rel}')
 a=yaml.safe_load(AUTH.read_text()); require(digest(AUTH)==AUTH_SHA and a['status']=='B02_GPU_FEATURE_PILOT_AUTHORIZED' and a['scope']['combined_shape']==[9020,2048],'authorization drift')
 require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS_WORKSPACE_CONFIG must be set before launch')
 require(not OUTPUT.exists() and not OUTPUT.is_symlink() and not list(OUTPUT.parent.glob('.BRCA_BATCH_B02.features.staging.*')),'output collision')
 require(git('rev-parse','HEAD',cwd=OFFICIAL)==OFFICIAL_HEAD and not git('status','--porcelain',cwd=OFFICIAL),'official HEALNet drift')
def inputs():
 require(not WSI.is_symlink() and WSI.is_file() and WSI.stat().st_size==WSI_SIZE,'WSI identity')
 require(digest(WSI,'md5')==WSI_MD5 and digest(WSI)==WSI_SHA,'WSI hashes')
 r=validate_brca_coordinate_artifacts(COORD,expected_manifest_sha256=COORD_SHA)
 require(r.branch_for('scale_2x').coordinate_count==7158 and r.branch_for('scale_4x').coordinate_count==1862,'coordinate counts')
 coords=[]
 for b in ('scale_2x','scale_4x'):
  with h5py.File(COORD/f'{b}_coordinates.h5','r') as h: coords.append(torch.from_numpy(np.asarray(h['coords'],dtype=np.int64)).contiguous())
 o=load_official_brca_patient_omics(OMIC,case_id=PATIENT,slide_id=SLIDE); require(o.source_row_index=='472' and digest(OMIC)==BRCA_RELEASE_ARCHIVE_SHA256,'Omic identity')
 require(CHECKPOINT.stat().st_size==102540417 and digest(CHECKPOINT)==CHECKPOINT_SHA,'checkpoint identity')
 with openslide.OpenSlide(str(WSI)) as s:
  require(tuple(s.level_dimensions)==DIMS and all(abs(a-b)<1e-10 for a,b in zip(s.level_downsamples,DS)),'header drift')
 return coords[0],coords[1],o
def cuda():
 torch.manual_seed(0); torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one CUDA GPU required; CPU fallback forbidden')
 require('Tesla T4' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0)==(7,5),'Tesla T4 capability 7.5 required')
 return torch.device('cuda:0')
def run(commit):
 started=datetime.now(timezone.utc); clock=time.perf_counter(); preflight(commit); c2,c4,o=inputs(); device=cuda()
 synthetic=run_one_patient_supervisor_healnet_smoke(official_repo=OFFICIAL,wsi=torch.zeros((1,9020,2048),device=device),rna=torch.zeros((1,1,1558),device=device),mutation=torch.zeros((1,1,21),device=device),cnv=torch.zeros((1,1,1333),device=device))
 model=build_resnet50_imagenet1k_v2(CHECKPOINT).to(device).eval()
 r2=extract_feature_matrix(StreamingOpenSlideDataset(WSI,PatchBranchSpec('scale_2x',c2,0,256)),model,device=device,batch_size=32,num_workers=2)
 r4=extract_feature_matrix(StreamingOpenSlideDataset(WSI,PatchBranchSpec('scale_4x',c4,1,256)),model,device=device,batch_size=32,num_workers=2)
 combined=torch.cat((r2.features,r4.features),dim=0).contiguous(); require(combined.shape==(9020,2048) and combined.dtype==torch.float32 and bool(torch.isfinite(combined).all()),'combined tensor')
 real=run_one_patient_supervisor_healnet_smoke(official_repo=OFFICIAL,wsi=combined.unsqueeze(0).to(device),rna=o.rna.to(device),mutation=o.mutation.to(device),cnv=o.cnv.to(device))
 rows=build_two_scale_provenance(scale_2x=BranchProvenanceSpec('scale_2x',c2,0,.501,.501),scale_4x=BranchProvenanceSpec('scale_4x',c4,1,1.002037431239458,1.002037431239458),scale_2x_count=7158,scale_4x_count=1862)
 meta=CompactFeatureMetadata(PATIENT,SLIDE,UUID,WSI_SHA,COORD_SHA,BRCA_RELEASE_ARCHIVE_SHA256,CHECKPOINT_SHA,POLICY_SHA,commit,7158,1862)
 manifest=publish_compact_feature_artifacts(OUTPUT,combined_features=combined,row_provenance=rows,metadata=meta,preserve_failed_staging=True); anchor=digest(OUTPUT/'compact_manifest.json'); validate_compact_feature_artifacts(OUTPUT,expected_manifest_sha256=anchor)
 return {'status':'BRCA_BATCH_B02_GPU_FEATURE_PILOT_SUCCESS','started_at_utc':started.isoformat(),'finished_at_utc':datetime.now(timezone.utc).isoformat(),'total_seconds':time.perf_counter()-clock,'source_commit':commit,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'torchvision':torchvision.__version__,'counts':{'scale_2x':7158,'scale_4x':1862,'total':9020},'feature_shapes':{'scale_2x':list(r2.features.shape),'scale_4x':list(r4.features.shape),'combined':list(combined.shape),'natural':[1,9020,2048]},'timings':{'scale_2x':r2.streaming_extraction_seconds,'scale_4x':r4.streaming_extraction_seconds},'peak_gpu_memory_bytes':max(r2.peak_gpu_memory_bytes,r4.peak_gpu_memory_bytes),'synthetic_healnet':asdict(synthetic),'real_healnet':asdict(real),'artifact':{'directory':str(OUTPUT),'manifest_sha256':anchor,'manifest':manifest,'total_file_bytes':sum(p.stat().st_size for p in OUTPUT.iterdir())},'operations':{'training':0,'backward':0,'optimizer_steps':0,'amp':0,'tf32':0,'cpu_fallback':0,'coordinate_generation':0,'deletions':0,'other_patients':0}}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--expected-source-commit',required=True)
 try: result=run(p.parse_args().expected_source_commit)
 except Exception as e: print(json.dumps({'status':'BLOCKED','error':f'{type(e).__name__}: {e}'},sort_keys=True)); return 1
 print(json.dumps(result,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
