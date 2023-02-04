import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.processors as proc

def main():
    data = fd.get_t1_data(9460, "Det1726")
    wave = data[0]["waveform"]["values"].nda[1]
    
    reset = np.max(data[0]["waveform"]["values"].nda[1])
    start = np.min(data[0]["waveform"]["values"].nda[1])
    print(reset)
    print(start)

    slope, intercept = proc.line_fit(wave, length=6500)
    print(slope)
    
    x = np.asarray([x for x in range(0,6500)])
    y = x*slope + intercept
    print((((reset-start)/(slope))*8)/(1e+9))
    plt.plot(wave)
    plt.plot(x, y)
    #plt.xlim(0,6500)
    #plt.plot(wf_trap)
    #plt.title("Example Waveform Breakdown")
    #plt.xlabel("us")
    #plt.ylabel("adc")
    plt.show()

if __name__ == "__main__":
    main()