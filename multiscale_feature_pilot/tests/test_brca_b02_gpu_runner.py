from pathlib import Path
import ast,hashlib,yaml
ROOT=Path(__file__).resolve().parents[2]; RUNNER=ROOT/'scripts/run_brca_b02_gpu_pilot.py'; AUTH=ROOT/'multiscale_feature_pilot/config/brca_b02_gpu_execution_authorization.yaml'
def test_authorization_exact_scope():
 a=yaml.safe_load(AUTH.read_text()); assert a['status']=='B02_GPU_FEATURE_PILOT_AUTHORIZED' and a['scope']['combined_shape']==[9020,2048]
 assert (a['scope']['scale_2x_patch_reads'],a['scope']['scale_4x_patch_reads'])==(7158,1862)
 assert hashlib.sha256(a['authorization_statement'].encode()).hexdigest()==a['authorization_statement_sha256'] and all(a['prohibited'].values())
def test_runner_b02_only_and_no_training_calls():
 s=RUNNER.read_text(); tree=ast.parse(s); assert s.count('StreamingOpenSlideDataset(WSI')==2 and 'BRCA_BATCH_B01' not in s
 calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}; assert not {'backward','step','train'}.intersection(calls)
def test_deterministic_gpu_and_compact_contract():
 s=RUNNER.read_text()
 for token in ('CUBLAS_WORKSPACE_CONFIG', 'torch.use_deterministic_algorithms(True)','allow_tf32=False',"torch.device('cuda:0')",'Tesla T4','preserve_failed_staging=True','combined.shape==(9020,2048)'): assert token in s
