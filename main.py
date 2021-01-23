# -*- coding: utf-8 -*-
"""
--------------------------------------
[TInf_PL2] 
Trabalho_Prático 1
--------------------------------------
Eduardo F. F. Cruz         2018285164
--------------------------------------
"""
import matplotlib.image as mplimg
import scipy.io.wavfile as spiiowf
import matplotlib.pyplot as plt
import numpy as np
import huffmancodec as hc

dataPath='data/' #global
    
def loadTxtData(filename):    
    f=open(dataPath+filename,'r')
    txt=list(f.read())
    f.close()
    return np.asarray(txt,dtype=np.str_)
def loadImgData(filename):
    return mplimg.imread(dataPath+filename)
def loadAudioData(filename):
    return spiiowf.read(dataPath+filename)
        
def flattenData(data):
    #convert multidimensional data array to unidimensional array 
    if(data.ndim>1):
        return data.flatten()     
    else: #if data array is already unidimensional
        return data
        
def getOccurrenceArray(data,alphabet): #return occurrences array 
    #Note: alphabet is always unidimensional 
    occurrences=np.zeros(len(alphabet))#create zeros array with len==len(alphabet)
    
    if (data.dtype=='<U1'): #text has to analyzed differently 
        for sample in data:
            occurrences[alphabet==ord(sample)]+=1
    elif (data.dtype=='<U2'):
        lenA=(90-65+1)+(122-97+1) #size of alphabet
        for sample in data:
            s=ord(sample[0])*lenA+ord(sample[1])
            occurrences[alphabet==s]+=1
    else:
        for sample in data:
            occurrences[sample]+=1
            
    return occurrences
    
def viewHistogram(alphabet,occurrences,title):
    plt.figure()
    plt.bar(alphabet,occurrences)
    plt.xlabel("Alphabet Symbols")
    plt.ylabel("Symbol Occurrence")
    plt.title(title)
    plt.tight_layout()

def getEntropy(nSamples,occurrences): #return entropy value and probability's array
    probability=occurrences[occurrences>0]/nSamples #calculate probability only for nonzero occurrence values
    
    return (-np.sum(probability*np.log2(probability))),probability #consider only nonzero probabilities
  
def huffmanCodeAverageLength(data,dataA,p): #returns average length and variance
    #Huffman Coding
    codec=hc.HuffmanCodec.from_data(data.flatten())
    symbolsList,lengthsList=codec.get_code_len() #get sub-set of Alphabet Symbols with respective Huffman codewords lengths 
    lengthsArray=np.array(lengthsList)
    
    l=np.sum(p*lengthsArray) #average length= P(a1)*length_c(a1)+P(a2)*length_c(a2)+...+P(an)*length_c(an)
    v=(np.sum(p*(lengthsArray-l)**2))/np.sum(p) #weighted sample variance (variancia ponderada)
    return l,v

def pairData(data,alphabet): #grouping source's data in pairs 
    dataLen=len(data)
    alphabetLen=len(alphabet)
    if(dataLen%2==1): #in case dataLen is odd..ignore last element from data
        data=data[:dataLen-1]
        
    if (data.dtype=='<U1'):
        ##group source samples as pairs
        pairedData=np.char.add(data[::2],data[1::2])
        ##generate alphabet with paired symbols
        pairedAlphabet=np.concatenate([np.arange(65*alphabetLen+65,90*alphabetLen+90+1,dtype=np.uint16),np.arange(97*alphabetLen+97,122*alphabetLen+122+1,dtype=np.uint16)]) #ASCII from 65-90 + from 97-122

    else:
        ##group source samples as pairs
        pairedData=data[::2]*alphabetLen+data[1::2]
        ##generate alphabet with paired symbols     
        pairedAlphabet=np.arange(0,alphabetLen**2,dtype=np.uint16) #bottom 0 because source datatype is always unsigned int8 (if not char type)
            
    return pairedData,pairedAlphabet
    
def getWindowMutualInfo(queryData,queryEntropy,windowData,alphabet): #returns mutual information value between query and window's target data 
    #notice that queryData and windowData always have the same length
    alphabetLen=len(alphabet)
    #calculate window's entropy
    windowO=getOccurrenceArray(windowData,alphabet)
    windowEntropy,p=getEntropy(len(windowData),windowO)
    
    conjuntasO=np.zeros((alphabetLen,alphabetLen))#matriz de ocurrences conjuntas
    for i in range(len(queryData)):
        conjuntasO[queryData[i],windowData[i]]+=1 #increment occurrence of (queryData[i],windowData[i])
    
    conjuntaEntropy,p=getEntropy(len(queryData),conjuntasO) #no need to flatten windowO. len(queryData) is equal to the number of samples conjuntas
    
    return queryEntropy+windowEntropy-conjuntaEntropy #informacao mutua para query e janela 

def getMutualInfoArray(queryData,targetData, alphabet, step):
    #query
    queryLen=len(queryData) #number of samples query's audio
    queryO=getOccurrenceArray(queryData,alphabet)
    queryEntropy,p=getEntropy(len(queryData),queryO)
    #target
    targetLen=len(targetData) #number of samples of target's audio
    #target's window qnt
    windowsQnt=int(np.floor((targetLen-queryLen)/step)+1)
    
    #array to save mutual info for each window     
    mutualInfoArray=np.zeros(windowsQnt)
    
    for i in range(0,windowsQnt):
        mutualInfoArray[i]=getWindowMutualInfo(queryData,queryEntropy,targetData[i*step:queryLen+i*step],alphabet) 

    return mutualInfoArray

def plotMutualInfo(t,mutualInfo,title):
    plt.figure()
    plt.plot(t,mutualInfo) 
    plt.xlabel("Time (s)")
    plt.ylabel("Mutual Information")
    plt.title(title)
    
def musicIdentifier(query,alphabet,step,queryName): 
    maxArr=np.zeros(7) #7 == quantity of targets to be analyzed
    for i in range(1,8):
        targetName='Song0'+str(i)+'.wav'
        [fs,target]= loadAudioData(targetName) 
        if(target.ndim>1): #if audio is stereo consider left channel
            target=target[:,0] 
        mutualInfo=getMutualInfoArray(query,target,alphabet,step) #get array with mutual information 
        print('\n[{} ; {}]:'.format(queryName,targetName))
        print('Evolução da Inf. Mútua: ',end='')
        print(mutualInfo) #print array with mutual info per sliding window
        maxArr[i-1]=np.max(mutualInfo) 
        print('Informação Mútua Máxima = {} '.format(maxArr[i-1]))
        #plotMutualInfo(t,mutualInfo,'Inf({};{})='.format(queryName,targetName))
    
    #print conclusion results
    sortedMaxArr=np.sort(maxArr)[::-1]   #descending order     
    print('\n'+5*'='+'Results for '+queryName+5*'=')
    for inf in sortedMaxArr:
        i=np.where(maxArr==inf)[0][0]+1
        targetName='Song0'+str(i)+'.wav'
        print('[{}]: Informação Mútua Máxima de {} bits/símbolo'.format(targetName,inf))
    print(35*'=')            
    
def main():  
    #==================================================
    #image
    print(50*'-') 
    filename='lena.bmp'
    tag='['+filename+']'
    lenaData=loadImgData(filename) #load data from file 
    nBits=int(str(lenaData.dtype)[4:]) #get number of bits from data type
    lenaData=flattenData(lenaData) #make data unidimensional
    lenaA=np.arange(0,2**nBits,dtype=lenaData.dtype) #img alphabet from 0 to (2**nBits)-1. Note: arange(start,end[not inclusive],step) => end=2**nBits 
    lenaO=getOccurrenceArray(lenaData,lenaA) #calculate symbol occurrence array and source's number of samples
    print(tag+': {} bits/symbol'.format(nBits)) #print number of bits from data type  
    # using the huffman codec..calculates and prints average number of bits per symbol + variance of codewords lengths (bits/symbol) 
    viewHistogram(lenaA,lenaO,tag+': Symbol\'s Occurrence Histogram')  #plots histogram
    entropy,probabilities=getEntropy(len(lenaData),lenaO)
    print(tag+': Entropy for this source = {} bits/symbol'.format(entropy))
    #ex4
    l,v=huffmanCodeAverageLength(lenaData,lenaA,probabilities)
    print(tag+': Huffman Codec: average of {} bits/symbol'.format(l))
    print(tag+': Huffman Codec: variance = {}'.format(v))
    print(tag+': Redundancy of the Huffman Code = {} bits/symbol'.format(l-entropy)) #a measure of the efficiency of this code (Huffman) is its redundancy
    #ex5
    lenaPairedData,lenaPairedA=pairData(lenaData,lenaA) #get source's samples and alphabet's symbols, grouped in pairs
    lenaPairedO=getOccurrenceArray(lenaPairedData,lenaPairedA)
    entropy,probabilities=getEntropy(len(lenaPairedData),lenaPairedO)
    print(tag+': Entropy for this GROUPED source = {} bits/symbol'.format(entropy/2))
    print(50*'-') 
    #------------------------------------------------
    filename='ct1.bmp'
    tag='['+filename+']'
    ct1Data=loadImgData(filename)
    nBits=int(str(ct1Data.dtype)[4:])
    ct1Data=flattenData(ct1Data[:,:,0]) #make data unidimensional
    ct1A=np.arange(0,2**nBits,dtype=ct1Data.dtype) #img alphabet from 0 to (2**nBits)-1
    ct1O=getOccurrenceArray(ct1Data,ct1A) #ct1 is a monocromatic image with rgb+alpha channels, so rgb channels will have the same value. For that reason, we'll only consider the r [0] channel values for efficiency purposes 
    print(tag+': {} bits/symbol'.format(nBits))
    viewHistogram(ct1A,ct1O,tag+': Symbol\'s Occurrence Histogram')  
    entropy,probabilities=getEntropy(len(ct1Data),ct1O)
    print(tag+': Entropy for this source = {} bits/symbol'.format(entropy))
    #ex4
    l,v=huffmanCodeAverageLength(ct1Data,ct1A,probabilities)
    print(tag+': Huffman Codec: average of {} bits/symbol'.format(l))
    print(tag+': Huffman Codec: variance = {}'.format(v))
    print(tag+': Redundancy of the Huffman Code = {} bits/symbol'.format(l-entropy)) #a measure of the efficiency of this code (Huffman) is its redundancy
    #ex5
    ct1PairedData,ct1PairedA=pairData(ct1Data,ct1A)
    ct1PairedO=getOccurrenceArray(ct1PairedData,ct1PairedA)
    entropy,probabilities=getEntropy(len(ct1PairedData),ct1PairedO)
    print(tag+': Entropy for this GROUPED source = {} bits/symbol'.format(entropy/2))
    print(50*'-') 
    #-------------------------------------------------
    filename='binaria.bmp'
    tag='['+filename+']'
    binariaData=loadImgData(filename)
    nBits=int(str(binariaData.dtype)[4:])
    binariaData=flattenData(binariaData[:,:,0])
    binariaA=np.arange(0,2**nBits,dtype=binariaData.dtype) #img alphabet from 0 to (2**nBits)-1 
    binariaO=getOccurrenceArray(binariaData,binariaA) #binaria is a binary image with 4 channels. There's only rgb(0,0,0) and rgb(255,255,255) colors. For that reason, similarly to ct1 image, we'll only consider the r [0] channel values for efficiency purposes 
    print(tag+': {} bits/symbol'.format(nBits))
    viewHistogram(binariaA,binariaO,tag+': Symbol\'s Occurrence Histogram')  
    entropy,probabilities=getEntropy(len(binariaData),binariaO)
    print(tag+': Entropy for this source = {} bits/symbol'.format(entropy))
    #ex4
    l,v=huffmanCodeAverageLength(binariaData,binariaA,probabilities)
    print(tag+': Huffman Codec: average of {} bits/symbol'.format(l))
    print(tag+': Huffman Codec: variance = {}'.format(v))
    print(tag+': Redundancy of the Huffman Code = {} bits/symbol'.format(l-entropy)) #a measure of the efficiency of this code (Huffman) is its redundancy
    #ex5
    binariaPairedData,binariaPairedA=pairData(binariaData,binariaA)
    binariaPairedO=getOccurrenceArray(binariaPairedData,binariaPairedA)
    entropy,probabilities=getEntropy(len(binariaPairedData),binariaPairedO)
    print(tag+': Entropy for this GROUPED source = {} bits/symbol'.format(entropy/2))
    print(50*'-') 
    #==================================================
    #audio
    filename='saxriff.wav'
    tag='['+filename+']'
    [saxriffFs,saxriffData]= loadAudioData(filename)
    nBits=int(str(saxriffData.dtype)[4:]) #bits quantization (unsigned 8 bits integer =>sample values from 0 to 2**bitsQ )
    saxriffData=flattenData(saxriffData[:,0])
    saxriffA=np.arange(0,2**nBits,dtype=saxriffData.dtype)
    saxriffO=getOccurrenceArray(saxriffData,saxriffA) 
    print(tag+': {} bits/symbol'.format(nBits))
    viewHistogram(saxriffA,saxriffO,tag+': Symbol\'s Occurrence Histogram') 
    entropy,probabilities=getEntropy(len(saxriffData),saxriffO)
    print(tag+': Entropy for this source = {} bits/symbol'.format(entropy))
    #ex4
    l,v=huffmanCodeAverageLength(saxriffData,saxriffA,probabilities)
    print(tag+': Huffman Codec: average of {} bits/symbol'.format(l))
    print(tag+': Huffman Codec: variance = {}'.format(v))
    print(tag+': Redundancy of the Huffman Code = {} bits/symbol'.format(l-entropy)) #a measure of the efficiency of this code (Huffman) is its redundancy
    #ex5
    saxriffPairedData,saxriffPairedA=pairData(saxriffData,saxriffA)
    saxriffPairedO=getOccurrenceArray(saxriffPairedData,saxriffPairedA)
    entropy,probabilities=getEntropy(len(saxriffPairedData),saxriffPairedO)
    print(tag+': Entropy for this GROUPED source = {} bits/symbol'.format(entropy/2))
    print(50*'-') 
    #==================================================
    #text
    filename='texto.txt'
    tag='['+filename+']'
    textoData=loadTxtData(filename)
    textoData=textoData[((textoData>='a')&(textoData<='z'))|((textoData>='A')&(textoData<='Z'))] #since we're not considering ascii letters outside [A-Za-z]..we remove them from textoData for efficiency purposes (since we have to pass the data without those 'special' characters, as argument, to the Huffman codec later)
    textoData=flattenData(textoData)
    textoA=np.concatenate([np.arange(65,91,dtype=np.uint8),np.arange(97,123,dtype=np.uint8)]) #texto Alphabet filled with decimal ascii values(8bits) of chars [A-Za-z]
    textoO=getOccurrenceArray(textoData,textoA)
    print(tag+': ASCII = 8 bits/symbol')
    viewHistogram(textoA,textoO,tag+': Symbol\'s Occurrence Histogram')
    entropy,probabilities=getEntropy(len(textoData),textoO)
    print(tag+': Entropy for this source = {} bits/symbol'.format(entropy))
    #ex4
    l,v=huffmanCodeAverageLength(textoData,textoA,probabilities) 
    print(tag+': Huffman Codec: average of {} bits/symbol'.format(l))
    print(tag+': Huffman Codec: variance = {}'.format(v))
    print(tag+': Redundancy of the Huffman Code = {} bits/symbol'.format(l-entropy)) #a measure of the efficiency of this code (Huffman) is its redundancy
    #ex5
    textoPairedData,textoPairedA=pairData(textoData,textoA)
    textoPairedO=getOccurrenceArray(textoPairedData,textoPairedA)
    entropy,probabilities=getEntropy(len(textoPairedData),textoPairedO)
    print(tag+': Entropy for this GROUPED source = {} bits/symbol'.format(entropy/2))
    #Note: dtype of textoData is Unicode. The ASCII value of a character is the same as its Unicode value.
    print(50*'-') 
    #---------------------------------------------------
    #ex6
    #TESTE de informacao mutua q está no enunciado do trabalho
    '''
    q=np.array([2 ,6 ,4 ,10 ,5 ,9, 5, 8 ,0 ,8],dtype=np.uint8)
    t=np.array([6, 8, 9, 7, 2, 4, 9 ,9 ,4 ,9 ,1 ,4 ,8 ,0 ,1 ,2 ,2 ,6 ,3 ,2 ,0, 7, 4, 9, 5, 4, 8, 5, 2, 7, 8 ,0 ,7 ,4 ,8 ,5 ,7 ,4 ,3 ,2 ,2 ,7 ,3 ,5,2, 7, 4, 9, 9, 6],dtype=np.uint8)
    a=np.arange(0,11,dtype=np.uint8)
    print('TESTE:')
    print(getMutualInfoArray(q,t,a,1))
    print('FIM TESTE')
    '''
    step=round(len(saxriffData)/4) #for 6b) and 6c)
    #6b)
    [fs,target1Data]= loadAudioData('target01 - repeat.wav')
    mutualInfo1=getMutualInfoArray(saxriffData,flattenData(target1Data[:,0]),saxriffA,step)
    print('\n[saxriff.wav ; target01-repeat.wav]')
    print('Evolução da Inf. Mútua :',end='')
    print(mutualInfo1)
    #plot
    t=np.arange(0,(target1Data.shape[0]-len(saxriffData)+1)/fs,step/fs) #calculate time instance array (consider beggining of the sliding window instance)
    plotMutualInfo(t,mutualInfo1,'[saxriff.wav ; target01-repeat.wav]')
       
    [fs,target2Data]= loadAudioData('target02 - repeatNoise.wav')
    mutualInfo2=getMutualInfoArray(saxriffData,flattenData(target2Data[:,0]),saxriffA,step)
    print('\n[saxriff.wav ; target02-repeatNoise.wav]')
    print('Evolução da Inf. Mútua :',end='')
    print(mutualInfo2) 
    #plot
    t=np.arange(0,(target2Data.shape[0]-len(saxriffData)+1)/fs,step/fs) #calculate time instance array (consider beggining of the sliding window instance)
    plotMutualInfo(t,mutualInfo2,'[saxriff.wav ; target02-repeatNoise.wav]')
    
    #6c)
    print(50*'-') 
    musicIdentifier(saxriffData,saxriffA,step,'saxriff.wav')
        
    #===============================THE END======================================
    
if __name__=="__main__":
    main()