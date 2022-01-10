#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import numpy as np
import pywt
import logging
import struct
import zlib
import math

import minimal
from temporal_no_overlapped_DWT_coding import Temporal_No_Overlapped_DWT as temp_no_DWT
from stereo_MST_coding_32 import Stereo_MST_Coding_32 as stereo32
from temporal_no_overlapped_DWT_coding import Temporal_No_Overlapped_DWT__verbose as temp_no_DWT__verbose

class Temporal_Overlapped_DWT(temp_no_DWT):
    
    def __init__(self):
        super().__init__()
        logging.info(__doc__)
        
        self.overlaped_area_size = self.max_filters_length * (1 << self.dwt_levels)

        # Mantenemos los chunks durante el Encoder
        self.list = [] 
        for i in range(3):
            self.list.append(np.zeros((minimal.args.frames_per_chunk, self.NUMBER_OF_CHANNELS), dtype=np.int32))
            
        # Mantenemos los chunks durante el Decoder
        self.decList = [] 
        for i in range(3):
            self.decList.append(np.zeros((minimal.args.frames_per_chunk, self.NUMBER_OF_CHANNELS), dtype=np.int32))
        
       
        #Cálculo de subbandas para 1024      
        self.subband = []
        aux =pywt.wavedecn_shapes((1024,), wavelet= self.wavelet, level=self.dwt_levels, mode='per')
        
        self.subband.extend(aux[0])
        for i in range(1,len(aux)):
            self.subband.extend(list(aux[i].values()))
            self.subband[i] = self.subband[i][0]
        
        # Cálculo de subbandas
        self.bandas = []
        aux =pywt.wavedecn_shapes((1024+2*self.overlaped_area_size,), wavelet=self.wavelet, level=self.dwt_levels, mode='per')
        
        self.bandas.extend(aux[0])
        for i in range(1,len(aux)):
            self.bandas.extend(list(aux[i].values()))
            self.bandas[i] = self.bandas[i][0]           
    
        
        #Coeficientes para extendido
        zero_array = np.zeros(shape=minimal.args.frames_per_chunk+2*self.overlaped_area_size)
        coeffs = pywt.wavedec(zero_array, wavelet=self.wavelet, level=self.dwt_levels, mode="per")
        self.slices = pywt.coeffs_to_array(coeffs)[1]
        

        
    # Se trata del Encode sección 1.2

    def analyze(self,chunk):

        # Input C_i+1          
        self.list[2] = chunk

        # Build extended chunk
        e = np.concatenate((self.list[0][-self.overlaped_area_size:], self.list[1], self.list[2][:self.overlaped_area_size]))

        # Compute extended decomposition
        d= self.extended_analyze(e)
        
       
        # Decomposition subset       
        subset_d = d[0: self.bandas[0]][self.overlaped_area_size//2**self.dwt_levels : -self.overlaped_area_size//2**self.dwt_levels]
                    
        variable= self.bandas[0]
        for i in range(self.dwt_levels):
            subset_d = np.concatenate((subset_d,d[variable : variable+self.bandas[i+1]][self.overlaped_area_size//2**(self.dwt_levels-i) : -self.overlaped_area_size//2**(self.dwt_levels-i)]))
            variable += self.bandas[i+1]
            
        # C_i-1 <-- C_i
        self.list[0] = self.list[1]
        # C_i <-- C_i+1
        self.list[1] = self.list[2]
        return subset_d


    # Compute extended decomposition
    def extended_analyze(self, chunk):
        chunk = stereo32.analyze(self,chunk)
        DWT_chunk = np.zeros((minimal.args.frames_per_chunk+2*self.overlaped_area_size, self.NUMBER_OF_CHANNELS), dtype=np.int32)
        for c in range(self.NUMBER_OF_CHANNELS):
            channel_coeffs = pywt.wavedec(chunk[:, c], wavelet=self.wavelet, level=self.dwt_levels, mode="per")
            channel_DWT_chunk = pywt.coeffs_to_array(channel_coeffs)[0]
            DWT_chunk[:, c] = channel_DWT_chunk
        return DWT_chunk
    
    # Se trata del Decode sección 1.2
    def synthesize(self, chunk_DWT):
        
        # Input D_i+1
        self.decList[2] = chunk_DWT
        
        # Build extended decomposition
        divide = int(self.overlaped_area_size/(2**(self.dwt_levels-(i-1))))
        extendido = np.array(self.decList[0][0: self.subband[0]][-divide: ])
        acumulado = 0
        
        for i in range (self.dwt_levels):
                divide = int(self.overlaped_area_size/(2**(self.dwt_levels-(i-1))))
                extendido = np.concatenate((extendido,self.decList[0][acumulado : acumulado+self.subband[i]][-divide : ]))
                extendido = np.concatenate((extendido,self.decList[1][acumulado : acumulado+self.subband[i]]))
                extendido = np.concatenate((extendido,self.decList[2][acumulado : acumulado+self.subband[i]][ : divide]))
                acumulado += self.subband[i]

        # Compute extended chunk
        chunk = self.extended_synthesize(extendido)
        

        chunkDWT = chunk[self.overlaped_area_size : -self.overlaped_area_size]


        # D_i-1 <-- D_i
        self.decList[0] = self.decList[1]
        # D_i <-- D_i+1
        self.decList[1] = self.decList[2]
        
        return chunkDWT
      
    def extended_synthesize(self, chunk_DWT):
        chunk = np.zeros((minimal.args.frames_per_chunk+2*self.overlaped_area_size, self.NUMBER_OF_CHANNELS), dtype=np.int32)          
        for c in range(self.NUMBER_OF_CHANNELS):
            channel_coeffs = pywt.array_to_coeffs(chunk_DWT[:, c], self.slices, output_format="wavedec")
            chunk[:, c] = pywt.waverec(channel_coeffs, wavelet=self.wavelet, mode="per")
        chunk= stereo32.synthesize(self,chunk)
        return chunk   


'''
    # Ignores overlapping
    def synthesize(self, chunk_DWT):
        chunk = np.empty((minimal.args.frames_per_chunk, self.NUMBER_OF_CHANNELS), dtype=np.int32)
        for c in range(self.NUMBER_OF_CHANNELS):
            channel_coeffs = pywt.array_to_coeffs(chunk_DWT[:, c], self.slices, output_format="wavedec")
            chunk[:, c] = pywt.waverec(channel_coeffs, wavelet=self.wavelet, mode="per")
        chunk = Stereo_Coding.synthesize(self,chunk)
        return chunk
    '''
    
   
class threshold__verbose(threshold,temp_DWT__verbose):
    pass
try:
    import argcomplete  
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
        intercom = threshold()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()