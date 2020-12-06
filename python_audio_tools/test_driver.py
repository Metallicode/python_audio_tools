from python_audio_tools import AudioTools

#Testing
if __name__ == "__main__":
    at = AudioTools()
    x = at.MakeSignal( shape="saw", frequency=617, length=1.0)
    # x = at.ModulationEnvelop(steps=20, low_value=300, high_value=1800)
    # x = at.ModulationEnvelop(steps=20, low_value=300, high_value=1800, mod_type="logarithmic", args={'log':1.0})

    x = at.FilterSweep(x,hi_freq=2000)




    # x = at.KikGenerator(length=1.0, max_pitch=4000, min_pitch=10, log=20)
    # x = at.SnerGenerator(length=2.0, high_pitch=1300, low_pitch=170, log=3, mix=0.7)
    #x = at.CymbelGenerator( length=1.0, op_a_freq=5134,op_b_freq=840, noise_env=20, tone_env=5,cutoff=1700, mix=0.4)
    #x = at.Clip(x, thd=0.1)



    # x=at.OpenFile("f")
    


    
##    x = at.WavtableFromSample(x)

    # x=at.Echo(x,130, 5, mix=0.5,cutoff =7000)
    
    # x = at.Chorus(x, mod_frequency=0.05,depth=200)
    
####Test Signal generators
    
##    x = at.MakeSignal()
##    x = at.MakeSignal( shape="sin", frequency=4, length=2.0) 
##    x = at.MakeSignal( shape="triangle", frequency=4, length=2.0)
#   x = at.MakeSignal( shape="saw", frequency=60, length=2.0)
##    x = at.MakeSignal( shape="square", frequency=4, length=2.0)
 ##   x = at.MakeSignal( shape="random_noise", length=2.0)
  #  x = at.MakeSignal( shape="normal_noise", length=2.0)

####test FastFm
##    x = at.FastFM(m_frequency=2, c_frequency=440, length=10.0)

####test FM
##    x = at.MakeSignal( shape="saw", frequency=50, length=2.0)
    #x = at.FM(x,c_frequency=400,depth=10000.0 )


####test AM
##    x = at.MakeSignal( shape="triangle", frequency=60, length=10.0)
    # y = at.MakeSignal( shape="sin", frequency=80, length=2.0)
    # x = at.AM(x,y)
    # x = at.Clip(x, thd=0.03)
####test mix
##    x = at.MakeSignal( shape="sin", frequency=550, length=2.0) 
##    y = at.MakeSignal( shape="triangle", frequency=170, length=2.0)
##    z = at.MakeSignal( shape="saw", frequency=90, length=2.0)
##    x = at.MixSignals(x,y,z)

  
##  x=at.Quantization(x, 0.28)
##
##    
##
##    

##    at.PlotFrequencyDomain(x, zoom=10)
    # at.PlotTimeDomain(x)
    at.MakeFile(x)