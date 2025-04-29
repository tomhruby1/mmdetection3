## load and dump config to a file or terminal

import sys
from pathlib import Path
from mmengine import Config

if __name__=='__main__':
    if len(sys.argv) in [2,3]:
        in_p = Path(sys.argv[1]).resolve()
        conf = Config.fromfile(in_p)
        if len(sys.argv) == 2:
            print(conf.dump())
        else:
            out_p = Path(sys.argv[2]).resolve()
            conf.dump(out_p)
    else:
        print("usage dump_config.py [input config path] [output dump config path]")        
