from pathlib import Path
import ast,hashlib,importlib.util,yaml
ROOT=Path(__file__).resolve().parents[2]; RUNNER=ROOT/'scripts/run_brca_b02_gpu_pilot.py'; PREP=ROOT/'multiscale_feature_pilot/config/brca_b02_gpu_preexecution.yaml'
def load_runner():
    spec=importlib.util.spec_from_file_location('b02_locked',RUNNER); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def test_preexecution_record_is_locked_and_bound():
    p=yaml.safe_load(PREP.read_text()); assert p['executable'] is False and not any(p['execution_lock'].values())
    assert hashlib.sha256(p['preparation_authority']['exact_statement'].encode()).hexdigest()==p['preparation_authority']['exact_statement_sha256']
    assert hashlib.sha256(p['future_exact_authorization_text'].encode()).hexdigest()==p['future_exact_authorization_text_sha256']
    assert p['inputs']['combined_shape']==[9020,2048]
def test_runner_first_runtime_gate_blocks_before_execute(monkeypatch):
    m=load_runner(); called=[]; monkeypatch.setattr(m,'_execute',lambda _:called.append(True))
    try: m.run('0'*40)
    except m.ExecutionLocked: pass
    else: raise AssertionError('runner did not lock')
    assert called==[] and m.EXECUTION_AUTHORIZED is False
def test_runner_exact_b02_contract_and_no_training_calls():
    s=RUNNER.read_text(); tree=ast.parse(s)
    assert '7158' in s and '1862' in s and 'TOTAL=9020' in s and 'BRCA_BATCH_B02.features' in s
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'backward','step','train'}.intersection(calls)
    for token in ('allow_tf32=False','preserve_failed_staging=True',"torch.device('cuda:0')",'CUBLAS_WORKSPACE_CONFIG'): assert token in s
