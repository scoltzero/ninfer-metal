#!/usr/bin/env python3
"""Self-contained generated current route trace + actual schema/router checks."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
exporter,checker,fixtures,libraries=map(Path,sys.argv[1:])
with tempfile.TemporaryDirectory(prefix='omlx-p12-tf-') as directory:
    root=Path(directory);source=root/'input.json';trace=root/'trace.json'
    source.write_text(json.dumps({'origin':'P12-TF control trace','synthetic':True,'sampling_contract':'position_coupled_v1','tokens':[(71*i+7)%257 for i in range(8)]}))
    model=fixtures/'moe-affine'
    subprocess.run([str(exporter),str(model/'source'),str(model/'model.omlxm5'),str(libraries),str(source),'0',str(trace)],check=True)
    subprocess.run([str(checker),str(trace),str(libraries)],check=True)
