




def read_functional(filename):
    """
    Require the tag of f= followed by value 
    
   
    """
    import numpy as np
    array=[]
    legend=[0.0,0.0,0.0,0.0]
    with open(filename) as f:
         lines = f.readlines()
         for line in lines:

             if "f=" in line:
                 array.append(float(line[23:35].replace("D","e")))
             elif "Namespace" in line:
 
                 legend[0] =line[line.rfind("alpha")+6:line.find("," ,line.find("alpha"))]
                 legend[1] =line[line.rfind("beta")+5:line.find("," ,line.find("beta"))]
                 legend[2] =line[line.rfind("k=")+2:line.find("," ,line.find("k="))]
                 legend[3] =len(eval(line[line.rfind("tau=")+4:line.find("]" ,line.find("tau="))+1]))

                 print legend
    return np.array(array),legend



def store_last_val(array,legend):
    with open("list_end.txt", "a") as myfile:
          print legend[0],legend[1],legend[2], legend[3],len(array[:,0]),array[-1,0],array[-1,1],array[-1,2] 
          myfile.write("{} & {} &{} &{} & {}& {} & {} & {} \n ".format(legend[0],legend[1],legend[2], legend[3],len(array[:,0]),array[-1,0],array[-1,1],array[-1,2] ) )




def extract_float(string):
    beg = string.find("(")+1
    end = string.find(")")
    return float(string[beg:end])

def read_slurm(filename):

    """
    Require the tag of DiffusionCoeffients followed by an array 
    
   
    """
    import numpy as np
    array = []
    legend=[0.0,0.0,0.0,0.0]
    with open(filename) as f:
         lines = f.readlines()
         for line in lines:

             
             if "Coeffs-Iter" in line:
                 spl0 , spl1 ,spl2 ,spl3 = line.split("|")
                 
                 D1=extract_float(spl1)
                 D2=extract_float(spl2)
                 D3=extract_float(spl3) 
                 array.append([0,D1,D2,D3])

             if "Functional-value" in line:
                 array[-1][0]= float(line.split("|")[1])
             
             elif "Namespace" in line:
                 legend[0] =line[line.rfind("alpha")+6:line.find("," ,line.find("alpha"))]
                 legend[1] =line[line.rfind("beta")+5:line.find("," ,line.find("beta"))]
                 legend[2] =line[line.rfind("k=")+2:line.find("," ,line.find("k="))]
                 legend[3] =len(eval(line[line.rfind("tau=")+4:line.find("]" ,line.find("tau="))+1]))

                 print legend
    return np.array(array),legend






if __name__=='__main__':
	import argparse
        import sys
        import matplotlib.pyplot as plt
        import numpy as np
        import os 

        from matplotlib.font_manager import FontProperties

        fontP = FontProperties()
        fontP.set_size('small')

        

        for no,k in enumerate(["Functional","CSF","Grey","White"]):
            print no
            for filename in sorted(os.listdir(sys.argv[1])):
               if filename.endswith(".out"):
                  array,legend = read_slurm(sys.argv[1]+"/"+filename)

                  if int(legend[2])==10 and int(legend[3])==10: 
                    plt.plot(array[:,no],label=r"$\alpha=%s,\beta=%s,k=%s,\tau=%s$"%(legend[0],legend[1], legend[2],legend [3]) )
                    store_last_val(array,legend)

           # plt.legend()
            plt.title(k)
            
            plt.ylabel("Diffusion Coefficent")
            plt.xlabel("Num iterations")
            plt.savefig("Nosie0_%s.png"%k)
            plt.show()
            plt.clf()
            
      



        
