from subprocess import call
import ctypes

import os


def check_so(folder, n_file, extension, path = None):
    if not os.path.exists(folder + n_file + '.so'):
        comp_so(folder, n_file, extension, path)
    #endif
          
    return



def comp_so(folder, name, extension, path = None ):
    """
    folder  = stringa con l'indirizzo della cartella dove e' contenuto il file
    name    = nome del file senza .*
    extension = e' la parte .* del nome del file
    """

    if os.path.exists(folder + name + extension) and not os.path.exists(folder + name + '.so'):

        if path == None:
                print("compiling " + folder + name + extension)
                err = call(["nvcc","-Xcompiler","-fPIC","-shared",'-arch=compute_50', '-code=compute_50,sm_50', 
                            '-lcusparse', '-lcublas',
                            "-o", folder + name + '.so', folder + name + extension])
        else:
                print("compiling " + folder + name + extension + " con path")
                err = call(["nvcc","-Xcompiler","-fPIC","-shared",'-arch=compute_50', '-code=compute_50,sm_50', 
                            '-lcusparse', '-lcublas',
                            "-I", path,  "-o", folder + name + '.so', folder + name + extension])
        #endif

    #endif
                
    return