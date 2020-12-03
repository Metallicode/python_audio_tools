import numpy as np
import scipy
from scipy.io import wavfile
from scipy import signal as sgl
import scipy.fftpack as fftpk
import matplotlib.pyplot as plt
from datetime import datetime

class AudioTools:  
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    #normalize  vector to -1.0 , 1.0 range, with adjustable offset
    def _norm(self, data):
        min_v = min(data)
        max_v = max(data)
        offset = min_v+max_v
        data = data+(offset/2)
        data = np.array([((x-min_v) / (max_v-min_v)) for x in data])*2.0-1
        return data * ((max_v/min_v)*-1)

    #will return a Time Vector with a given length and sample size of 1.0/sample_rate
    def _timevector(self, length):       
        return np.arange(0,length,1.0/self.sample_rate)

    #mix args signals
    def MixSignals(self, *signals):
        mixed = np.zeros(len(signals[0]))    
        for i in range(len(signals)):
            mixed += signals[i]
            
        return self._norm(mixed)


    def Filter(self,signal, flt_type="low", cutoff=1000, pol=4, filter_algo="butter"):      
        w = cutoff/(self.sample_rate/2)
        if filter_algo == "butter":       
            a,b = sgl.butter(pol, w, flt_type, analog=False)
        else:
            a,b = sgl.bessel(pol, w, flt_type, analog=False)
    
        return sgl.filtfilt(a,b,signal)


    def Overdrive(self,signal, drive=5):     
        return self._norm([1-np.exp(-x*drive) if x > 0 else -1+np.exp(x*drive) for x in signal]) 

    def Clip(self,signal, thd=0.9):     
        return np.array([x/np.abs(x)-((x/np.abs(x))-(thd*x/np.abs(x))) if np.abs(x) > thd and x != 0 else x for x in signal])

    #Quantize a signal (bitreduction), q is max at 1.
    def Quantization(self,signal, q=0.3):
        return q* np.round(signal/q)
    
    def Chorus(self,signal, mod_frequency=1.0,depth=50.0):
        mod = self.MakeSignal(shape="sin", frequency=mod_frequency, length=len(signal)/self.sample_rate)*depth
        product = np.zeros_like(signal)

        for i in range(0, len(signal)):
            if i + int(mod[i])<len(product):
                product[i] = signal[i + int(mod[i])]

        product = sgl.savgol_filter(product, 51, 3)
        return self._norm(signal + product)

    def Echo(self, signal, bpm, feedback,mix=1.0,cutoff =4000):
        #create filter
        w = cutoff/(self.sample_rate/2)
        a,b = sgl.butter(4, w, "low", analog=False)

        #calc time in samples
        x = round(1/bpm * 60 * self.sample_rate)
        d = x

        #allocate memory
        product = np.zeros_like(signal, dtype='float64')
        
        for i in range(feedback):
            #create empty array in length of delay time X feedback iteration
            shift = np.zeros(d)
            #concatenate <signal - tail(d)> to empty array
            delay = np.concatenate([shift, signal[:-d]*(1.0/(i+1))])
            #increase shift size for next iteration
            d += x
            #mix product with filtered & delayed signal
            product += sgl.filtfilt(a,b,delay)

        return self._norm((product*mix)+signal)


    #Create a Signal with a given waveform, frequency and length
    def MakeSignal(self, shape="sin", frequency=440, length=1.0):
        
        signal = None
        t = self._timevector(length)  
        x = t * np.pi * 2 * frequency
        
        #shape selector
        if shape == "sin":
            signal = np.sin(x)        
        elif shape == "triangle":
            #signal = sgl.sawtooth(2 * np.pi * frequency * t, 0.5)
            signal = np.abs((x/np.pi-0.5)%2-1)*2-1
        elif shape == "saw":
            #signal = sgl.sawtooth(2 * np.pi * frequency * t)  
            signal = -((x/np.pi)%2)+1   
        elif shape == "square":
            #signal = (np.mod(frequency*t , 1) < 0.5)*2.0-1 
            signal = np.where(x/np.pi % 2 > 1, -1,1)    
        elif shape == "random_noise":
            signal = 1.0*np.random.random(int(length*self.sample_rate))
        elif shape == "normal_noise":
            signal = 1.0*np.random.randn(int(length*self.sample_rate))       
        else:
            pass
        return signal

    #modulate signals amplitude
    def AM(self, signal_a, signal_b):
        return self._norm(signal_a*signal_b)

    #generate FM signal from two sin waves
    def FastFM(self, m_frequency=5 , c_frequency=440,length=1.0):
        t = self._timevector(length)      
        mod = self.MakeSignal(shape="sin", frequency=m_frequency, length=10.0)
        signal = np.sin(2 * np.pi * c_frequency*mod)
        signal=self._norm(signal)
        return signal

    #generate a sin and modulate it with a signal
    def FM(self, signal, c_frequency=440, depth=100.0):
        carrier_frequency = c_frequency

        product = np.zeros_like(signal)
        phase = 0

        #FM the signal
        for n in range(0, len(signal)):
                #calc phase
                phase += signal[n] * np.pi * depth / self.sample_rate
                phase %= 2 * np.pi
                #calc carrier
                carrier = 2 * np.pi * carrier_frequency * (n / float(self.sample_rate))
                #modulate signals
                product[n] = np.cos(phase) * np.cos(carrier) - np.sin(phase) * np.sin(carrier)

        return self._norm(product)


    def WavtableFromSample(self, signal,length = 5.0, min_peak_heigth = 100):
        FFT, freqs = self.FFT(signal)
        
        indices = sgl.find_peaks(FFT[range(len(FFT)//20)], height=min_peak_heigth, width=2)[0]
        peaks = [freqs[x] for x in indices]
        wave_table = [np.roll(self.MakeSignal("sin", i , length), int(i%180)) for i in peaks]
        
        return self.MixSignals(*wave_table)




    def FFT(self,signal):
        FFT = abs(scipy.fft.fft(signal))
        freqs = fftpk.fftfreq(len(FFT), (1.0/self.sample_rate))
        return FFT, freqs
        
    #show waveform
    def PlotTimeDomain(self, signal):
        plt.plot(self._timevector(len(signal)/self.sample_rate),signal)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    #FFT the signal
    def PlotFrequencyDomain(self, signal, zoom=50):
        FFT, freqs = self.FFT(signal)
        plt.plot(freqs[range(len(FFT)//zoom)], FFT[range(len(FFT)//zoom)], color="black")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()

    #Save to mono wave file
    def MakeFile(self, signal):
        now = datetime.now()
        signal=self._norm(signal)
        signal *=32767
        signal = np.int16(signal)
        wavfile.write(f"file{now.strftime('%d-%m-%Y %H-%M-%S')}.wav", self.sample_rate, signal)

    def OpenFile(self, name):
        samplerate, signal = wavfile.read(f"{name}.wav")
        return self._norm(np.array(signal,dtype=np.float64))






















#Testing
if __name__ == "__main__":
    at = AudioTools()








    # x=at.OpenFile("f")
    


    
##    x = at.WavtableFromSample(x)

    # x=at.Echo(x,130, 5, mix=0.5,cutoff =7000)
    
    # x = at.Chorus(x, mod_frequency=0.05,depth=200)
    
####Test Signal generators
    
##    x = at.MakeSignal()
##    x = at.MakeSignal( shape="sin", frequency=4, length=2.0) 
##    x = at.MakeSignal( shape="triangle", frequency=4, length=2.0)
##   x = at.MakeSignal( shape="saw", frequency=4, length=2.0)
    x = at.MakeSignal( shape="square", frequency=4, length=2.0)
##    x = at.MakeSignal( shape="random_noise", frequency=4, length=2.0)
##    x = at.MakeSignal( shape="normal_noise", frequency=4, length=2.0)

####test FastFm
##    x = at.FastFM(m_frequency=2, c_frequency=440, length=10.0)

####test FM
##    x = at.MakeSignal( shape="saw", frequency=50, length=2.0)
##    x = at.FM(x,c_frequency=100,depth=2000.0 )


####test AM
##    x = at.MakeSignal( shape="triangle", frequency=60, length=10.0)
##    y = at.MakeSignal( shape="sin", frequency=200, length=10.0)
##    x = at.AM(x,y)

####test mix
##    x = at.MakeSignal( shape="sin", frequency=550, length=2.0) 
##    y = at.MakeSignal( shape="triangle", frequency=170, length=2.0)
##    z = at.MakeSignal( shape="saw", frequency=90, length=2.0)
##    x = at.MixSignals(x,y,z)

  
##    x=at.Quantization(x, 0.05)
##
##    
##
##    

##    at.PlotFrequencyDomain(x, zoom=10)
    at.PlotTimeDomain(x)
    # at.MakeFile(x)



