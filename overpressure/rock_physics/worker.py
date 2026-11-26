"""One-request rock-physics calculation worker; never imports Qt or torch."""
import contextlib
import io
import json
import os
import sys
import traceback
from . import calculate,save_result

os.environ['MPLBACKEND']='Agg'


def main():
    if hasattr(sys.stdin,'reconfigure'):sys.stdin.reconfigure(encoding='utf-8')
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8')
    try:
        request=json.load(sys.stdin)
        # Upstream numerical diagnostics must not corrupt the JSON line protocol.
        log=io.StringIO()
        with contextlib.redirect_stdout(log):
            if request.get('operation') in ('well_log','well_live'):
                from .well_model import run_workflow
                def progress(message):
                    print(json.dumps(dict(event='stage',message=message),ensure_ascii=False),file=sys.__stdout__,flush=True)
                result=run_workflow(request['scheme'],request['well'],progress,
                    preview_limit=request.get('preview_limit',240) if request.get('operation')=='well_live' else None)
            elif request.get('operation')=='well_preview':
                from .well_model import prepare_data
                import numpy as np
                import pandas as pd
                frame,meta=prepare_data(request['scheme'],request['well'])
                columns=['Depth','phi','sw','toc','temperature','pressure','matrix_density','vp','vs','rho','qc_reason']
                sample=frame[columns].head(100).replace([np.inf,-np.inf],np.nan).astype(object)
                sample=sample.where(pd.notna(sample),None)
                reasons=frame.loc[~frame.model_valid,'qc_reason'].value_counts().to_dict()
                result=dict(message=f'总样本 {len(frame)}；有效 {int(frame.model_valid.sum())}；无效 {int((~frame.model_valid).sum())}。预览前 100 行。\n'+json.dumps(reasons,ensure_ascii=False)+'\n'+'\n'.join(meta['warnings']),table=dict(columns=columns,rows=sample.values.tolist()))
            else:result=calculate(request['model_id'],request['parameters'])
        if log.getvalue():sys.stderr.write(log.getvalue())
        if request.get('operation')=='well_live':
            print(json.dumps(dict(event='live_preview',result=result),ensure_ascii=False,allow_nan=False),flush=True);return 0
        if request.get('operation')=='well_preview':
            print(json.dumps(dict(event='preview',result=result),ensure_ascii=False,allow_nan=False),flush=True);return 0
        path=save_result(result,request['output_dir'])
        print(json.dumps(dict(event='completed',path=str(path),result=result),ensure_ascii=False,allow_nan=False),flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps(dict(event='error',message=str(exc)),ensure_ascii=False),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
