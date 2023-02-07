import os

def main():
    files = [x for x in range(8725, 8808)]
    for file in files:
        command = "scp jlb1694@hcdata.phy.ornl.gov:/data7/coherent/data/gemini/daq/Run" + str(file) + ".gz /media/jlb1694/Gemini-1/data/" 
        os.system(command)

if __name__=="__main__":
    main()