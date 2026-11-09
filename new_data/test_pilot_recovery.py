import json
import pytest
from new_data.pilot_zip import record_failure


def test_pilot_failure_records_event_and_status(tmp_path):
    calls=[]
    def status(stage, **items): calls.append((stage,items))
    with pytest.raises(ValueError,match='changed'):
        with record_failure(tmp_path,status):
            raise ValueError('snapshot changed')
    assert calls[0][0]=='failed' and calls[0][1]['error']=='snapshot changed'
    event=json.loads((tmp_path/calls[0][1]['event_file']).read_text(encoding='utf8'))
    assert event['error_type']=='ValueError' and event['event']=='failed'
