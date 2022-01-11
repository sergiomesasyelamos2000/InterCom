#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import numpy as np
import logging
import math
import minimal
from temporal_overlapped_DWT_coding import Temporal_Overlapped_DWT
from temporal_overlapped_DWT_coding import Temporal_Overlapped_DWT__verbose


class Threshold(Temporal_Overlapped_DWT):
    def __init__(self):
        super().__init__()
        logging.info(__doc__)
       
        # Declaramos las frecuencias
        self.frecuencias = []
        for i in range(len(self.bandas)):
            self.frecuencias.append('0')
        
        # Declaramos la Cuantización
        self.cuantizacion = []
        for i in range(len(self.bandas)):
            self.cuantizacion.append('0')
        
        # Ancho de banda de la señal de audio
        self.frecuencia = 22050
        if(self.dwt_levels == 0):
            self.frecuencias[0] = self.frecuencias
        else:    
            # Niveles DWT en función de filtro paso bajo y alto
            for i in range(self.dwt_levels,-1,-1):
                if(i == self.dwt_levels):
                    # Calculamos los anchos de banda para las diferentes subandas
                    # ya que sabemos que es el ancho de banda de la señal de audio
                    # dividido entre 2,4,...
                    self.frecuencias[i] = self.frecuencia/2 
                    # Empleamos bandas extendidas
                elif(self.bandas[i] == self.bandas[i+1]):
                    self.frecuencias[i] = self.frecuencia
                else:
                    self.frecuencias[i] = self.frecuencia/2 
                self.frecuencia = self.frecuencia/2
            self.cuantizacion[i] = abs(int(3.64*(self.frecuencias[i]/1000)**(-0.8)-6.5*math.exp((-0.6)*(self.frecuencias[i]/1000-3.3)**2)+ 10**(-3)*(self.frecuencias[i]/1000)**4))

    def analyze(self,chunk):  
        chunk_DWT = super().analyze(chunk)
        return self.quantize(self,chunk_DWT)

    def synthesize(self, chunk_DWT):
        self.dequantize(self,chunk_DWT)
        return super().synthesize(chunk_DWT)

    def quantize(self, chunk):
        for i in range (len(self.subband)):
           chunk[0: self.subband[i]] = (chunk[0: self.subband[i]] / self.cuantizacion[i]).astype(np.int32)
        return chunk
        
    def dequantize(self, quantized_chunk):
        for i in range (len(self.subband)):
           quantized_chunk[0: self.subband[i]] = quantized_chunk[0: self.subband[i]] * self.cuantizacion[i]
        return quantized_chunk
        
    
class threshold__verbose(Threshold,Temporal_Overlapped_DWT__verbose):
    pass

try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")

if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        logging.warning("argcomplete not working :-/")
    minimal.args = minimal.parser.parse_known_args()[0]
    if minimal.args.show_stats or minimal.args.show_samples:
        intercom = threshold__verbose()
    else:
        intercom = Threshold()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()
