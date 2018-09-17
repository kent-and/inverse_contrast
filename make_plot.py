




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
          print legend[0],legend[1],legend[2], legend[3],len(array[:,0]),array[-1,1],array[-1,2],array[-1,3] 
          myfile.write("{} & {} &{} &{} & {}& {} & {} & {} \\\\ \n ".format(legend[0],legend[1],legend[2], legend[3],len(array[:,0]),array[-1,1],array[-1,2],array[-1,3] ) )




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
                 legend[0] =float(line[line.rfind("alpha")+6:line.find("," ,line.find("alpha"))])
                 legend[1] =float(line[line.rfind("beta")+5:line.find("," ,line.find("beta"))])
                 legend[2] =int(line[line.rfind("k=")+2:line.find("," ,line.find("k="))])
                 legend[3] =eval(line[line.rfind("tau=")+4:line.find("]" ,line.find("tau="))+1])

             
    return np.array(array),legend


def regular(legend):
    return legend[3]==[2.4, 4.8, 7.2, 9.6, 12.0, 14.4, 16.8, 19.2, 21.6, 24.0]

def hole(legend):
    return legend[3]==[1.2, 1.4, 1.8, 2.4, 3.6, 5.4, 7.6, 17.2, 18.6, 20.4, 22.4, 24.0]

def unevenly(legend):
    return legend[3]==[1.0, 1.2, 1.4, 2.8, 4.6, 9.6, 14.4, 18.6, 22.2, 24.0]

def range_of_interrest(legend):
    return (legend[1] >1.0e-3 and legend[1] <10.0 and legend[0] >1.0e-7 and legend[0] <=1.0e-3 ) 


if __name__=='__main__':
	import argparse
        import sys
        import matplotlib.pyplot as plt
        import numpy as np
        import os 

        from matplotlib.font_manager import FontProperties

        fontP = FontProperties()
        fontP.set_size('small')


        #
        

        for no,k in enumerate(["Functional","CSF","Grey","White"]):
            print no
            for filename in sorted(os.listdir(sys.argv[1])):
               if filename.endswith(".out"):
                  array,legend = read_slurm(sys.argv[1]+"/"+filename)
                  
                  if array.all() and legend and range_of_interrest(legend) and legend[2]<=11 and legend[2]>5 and regular(legend): 
                    print legend
                    print array, filename
                    if no==0:
                       plt.loglog(array[:,no],label=r"$\alpha=%s,\beta=%s,k=%s,\tau=%s$"%(legend[0],legend[1], legend[2],len(legend[3])) )
                       store_last_val(array,legend)
                    else:
                       plt.plot(array[:,no],label=r"$\alpha=%s,\beta=%s,k=%s,\tau=%s$"%(legend[0],legend[1], legend[2],len(legend[3])) )
             
                     

            plt.legend()
            plt.title(k)
            
            plt.ylabel("Diffusion Coefficent")
            plt.xlabel("Num iterations")
            plt.savefig("Nosie0_%s.png"%k)
            plt.show()
            plt.clf()
            
      



        
