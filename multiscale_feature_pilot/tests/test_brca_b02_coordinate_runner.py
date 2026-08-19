from pathlib import Path
import ast,hashlib,yaml
ROOT=Path(__file__).resolve().parents[2]; RUNNER=ROOT/'scripts/run_brca_b02_coordinate_gate.py'; AUTH=ROOT/'multiscale_feature_pilot/config/brca_b02_coordinate_execution_authorization.yaml'
def test_exact_authorization_and_read_tuple():
    a=yaml.safe_load(AUTH.read_text()); assert a['status']=='AUTHORIZED_B02_SINGLE_MASK_READ_AND_COORDINATE_PUBLICATION'
    assert a['authorized_read']=={'openslide_open_count':1,'read_region_count':1,'level_0_location':[0,0],'level':2,'size_at_level':[5580,4560]}
    assert hashlib.sha256(a['approval']['exact_statement'].encode()).hexdigest()==a['approval']['exact_statement_sha256']
def test_runner_contains_one_read_call_and_no_model_surface():
    s=RUNNER.read_text(); tree=ast.parse(s)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='read_region']
    assert len(calls)==1
    for token in ('torch','cuda','ResNet','HealNet','DataLoader'): assert token not in s
def test_runner_is_b02_only_and_zero_delete_publication():
    s=RUNNER.read_text(); assert 'BRCA_BATCH_B02' in s and 'BRCA_BATCH_B01' not in s
    assert 'renameat2' in s and 'rmtree' not in s and '.unlink(' not in s
