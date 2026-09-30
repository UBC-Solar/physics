# import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()
s = eng.genpath('~/UBC_Solar/vdx-simulation/steering')
eng.addpath(s, nargout=0)

[deltaL, deltaR] = eng.getTireAnglesFromYoke(np.int64(10), nargout = 1)
print(deltaL,deltaR)
eng.quit()