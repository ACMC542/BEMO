import pandas as pd
import numpy as np
import os

from numpy.fft import fft, fftfreq
from scipy.stats import kurtosis
from scipy.stats import entropy 
from scipy.signal import welch
from scipy.fftpack import fft

from featuretools.primitives import Max, Min, Median, Mode, Skew, Std, Mean

max = Max()
min = Min()
mean = Mean()
median = Median()
mode = Mode()
skew = Skew()
std = Std()
    
def calc_rotating_frequency(tachometer):
    N = len(tachometer)
    y = tachometer.to_list()
    T = 0.00002
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yf2 = 2.0/N * np.abs(yf[0:N//2])
    dados = {'freq':xf,'magnitude':yf2}
    dframe = pd.DataFrame(data=dados)
    df_descendente = dframe.sort_values('magnitude', ascending = False)
    freqRotacao = df_descendente['freq'].iloc[0]    

    return freqRotacao

def calc_spectral(signal):
    N = len(signal)
    y = signal.to_list()
    T = 0.00002
    yf = fft(y)
    freqs = np.abs(yf)
    kj1 = (N * freqs) / y
    spectral = np.mean(kj1)
    return spectral

def calculaFFT(signal, frequenciaRotacao):
    periodo = 0.00002
    nAmostras = len(signal)
    y = signal.to_list()
    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*periodo), nAmostras//2)
    yf2 = 2.0/nAmostras * np.abs(yf[0:nAmostras//2])

    
    dados = {'freq':xf,'magnitude':yf2}
    dframe = pd.DataFrame(data=dados)

    rf = dframe['magnitude'].where((dframe['freq'] > (frequenciaRotacao -5)) 
    & (dframe['freq'] < (frequenciaRotacao + 5)));
    rf2 = dframe['magnitude'].where((dframe['freq'] > (2 * frequenciaRotacao -5)) 
    & (dframe['freq'] < (2 * frequenciaRotacao + 5)));
    rf3 = dframe['magnitude'].where((dframe['freq'] > (3 * frequenciaRotacao -5)) 
    & (dframe['freq'] < (3 * frequenciaRotacao + 5)));

    amplitudeRf = rf.max()
    amplitude2Rf = rf2.max()
    amplitude3Rf = rf3.max()

    return amplitudeRf, amplitude2Rf, amplitude3Rf
    

def generate_features(paths):

    columns=['rotating_frequency','kurtosis_uba_axial','kurtosis_uba_radial','kurtosis_uba_tangencial','kurtosis_oba_axial',
            'kurtosis_oba_radial','kurtosis_oba_tangencial','entropy_uba_axial','entropy_uba_radial','entropy_uba_tangencial','entropy_oba_axial',
            'entropy_oba_radial','entropy_oba_tangencial','spectral_uba_axial','spectral_uba_radial','spectral_uba_tangencial','spectral_oba_axial',
            'spectral_oba_radial','spectral_oba_tangencial','h_rf_uba_axial', 'h_rf2_uba_axial', 'h_rf3_uba_axial','h_rf_uba_radial', 'h_rf2_uba_radial', 
            'h_rf3_uba_radial','h_rf_uba_tangencial','h_rf2_uba_tangencial','h_rf3_uba_tangencial','h_rf_oba_axial','h_rf2_oba_axial','h_rf3_oba_axial',
            'h_rf_oba_radial','h_rf2_oba_radial','h_rf3_oba_radial','h_rf_oba_tangencial', 'h_rf2_oba_tangencial','h_rf3_oba_tangencial','max_uba_axial',
            'min_uba_axial','mean_uba_axial','median_uba_axial','mode_uba_axial','skew_uba_axial','std_uba_axial','max_uba_radial','min_uba_radial','mean_uba_radial',
            'median_uba_radial','mode_uba_radial','skew_uba_radial','std_uba_radial','max_uba_tangencial','min_uba_tangencial','mean_uba_tangencial','median_uba_tangencial',
            'mode_uba_tangencial','skew_uba_tangencial','std_uba_tangencial','max_oba_axial','min_oba_axial','mean_oba_axial','median_oba_axial',
            'mode_oba_axial','skew_oba_axial','std_oba_axial','max_oba_radial','min_oba_radial','mean_oba_radial','median_oba_radial','mode_oba_radial',
            'skew_oba_radial','std_oba_radial','max_oba_tangencial','min_oba_tangencial','mean_oba_tangencial','median_oba_tangencial','mode_oba_tangencial',
            'skew_oba_tangencial','std_oba_tangencial','max_tachometer','min_tachometer','mean_tachometer','median_tachometer','mode_tachometer','skew_tachometer',
            'std_tachometer','max_microphone','min_microphone','mean_microphone','median_microphone','mode_microphone','skew_microphone','std_microphone']

    for path in paths:

        result = pd.DataFrame()

        res = os.listdir(path)
        for file in res:
            print(path + '/' + file)
            df = pd.read_csv(path + '/' + file, header=None, index_col=False, 
                names=['tachometer','uba_axial','uba_radial','uba_tangencial','oba_axial','oba_radial','oba_tangencial','microphone'])

            frequenciaRotacao = calc_rotating_frequency(df['tachometer'])
            
            features = {}

            features['rotating_frequency'] = frequenciaRotacao

            features['kurtosis_uba_axial'] = kurtosis(df['uba_axial'], fisher=False)
            features['kurtosis_uba_radial'] = kurtosis(df['uba_radial'], fisher=False)
            features['kurtosis_uba_tangencial'] = kurtosis(df['uba_tangencial'], fisher=False)
            features['kurtosis_oba_axial'] = kurtosis(df['oba_axial'], fisher=False)
            features['kurtosis_oba_radial'] = kurtosis(df['oba_radial'], fisher=False)
            features['kurtosis_oba_tangencial'] = kurtosis(df['oba_tangencial'], fisher=False)
            
            features['entropy_uba_axial'] = entropy(df['uba_axial'].value_counts())
            features['entropy_uba_radial'] = entropy(df['uba_radial'].value_counts())
            features['entropy_uba_tangencial'] = entropy(df['uba_tangencial'].value_counts())
            features['entropy_oba_axial'] = entropy(df['oba_axial'].value_counts())
            features['entropy_oba_radial'] = entropy(df['oba_radial'].value_counts())
            features['entropy_oba_tangencial'] = entropy(df['oba_tangencial'].value_counts())

            features['spectral_uba_axial'] = calc_spectral(df['uba_axial'])
            features['spectral_uba_radial'] = calc_spectral(df['uba_radial'])
            features['spectral_uba_tangencial'] = calc_spectral(df['uba_tangencial'])
            features['spectral_oba_axial'] = calc_spectral(df['oba_axial'])
            features['spectral_oba_radial'] = calc_spectral(df['oba_radial'])
            features['spectral_oba_tangencial'] = calc_spectral(df['oba_tangencial'])

            features['h_rf_uba_axial'], features['h_rf2_uba_axial'], features['h_rf3_uba_axial'] = calculaFFT(df['uba_axial'], frequenciaRotacao)
            features['h_rf_uba_radial'], features['h_rf2_uba_radial'], features['h_rf3_uba_radial'] = calculaFFT(df['uba_radial'], frequenciaRotacao)
            features['h_rf_uba_tangencial'], features['h_rf2_uba_tangencial'], features['h_rf3_uba_tangencial'] = calculaFFT(df['uba_tangencial'], frequenciaRotacao)
            features['h_rf_oba_axial'], features['h_rf2_oba_axial'], features['h_rf3_oba_axial'] = calculaFFT(df['oba_axial'], frequenciaRotacao)
            features['h_rf_oba_radial'], features['h_rf2_oba_radial'], features['h_rf3_oba_radial'] = calculaFFT(df['oba_radial'], frequenciaRotacao)
            features['h_rf_oba_tangencial'], features['h_rf2_oba_tangencial'], features['h_rf3_oba_tangencial'] = calculaFFT(df['oba_tangencial'], frequenciaRotacao)

            features['max_uba_axial'] = max(df['uba_axial'])
            features['min_uba_axial'] = min(df['uba_axial'])
            features['mean_uba_axial'] = mean(df['uba_axial'])
            features['median_uba_axial'] = median(df['uba_axial'])
            features['mode_uba_axial'] = mode(df['uba_axial'])
            features['skew_uba_axial'] = skew(df['uba_axial'])
            features['std_uba_axial'] = std(df['uba_axial'])

            features['max_uba_radial'] = max(df['uba_radial'])
            features['min_uba_radial'] = min(df['uba_radial'])
            features['mean_uba_radial'] = mean(df['uba_radial'])
            features['median_uba_radial'] = median(df['uba_radial'])
            features['mode_uba_radial'] = mode(df['uba_radial'])
            features['skew_uba_radial'] = skew(df['uba_radial'])
            features['std_uba_radial'] = std(df['uba_radial'])

            features['max_uba_tangencial'] = max(df['uba_tangencial'])
            features['min_uba_tangencial'] = min(df['uba_tangencial'])
            features['mean_uba_tangencial'] = mean(df['uba_tangencial'])
            features['median_uba_tangencial'] = median(df['uba_tangencial'])
            features['mode_uba_tangencial'] = mode(df['uba_tangencial'])
            features['skew_uba_tangencial'] = skew(df['uba_tangencial'])
            features['std_uba_tangencial'] = std(df['uba_tangencial'])

            features['max_oba_axial'] = max(df['oba_axial'])
            features['min_oba_axial'] = min(df['oba_axial'])
            features['mean_oba_axial'] = mean(df['oba_axial'])
            features['median_oba_axial'] = median(df['oba_axial'])
            features['mode_oba_axial'] = mode(df['oba_axial'])
            features['skew_oba_axial'] = skew(df['oba_axial'])
            features['std_oba_axial'] = std(df['oba_axial'])

            features['max_oba_radial'] = max(df['oba_radial'])
            features['min_oba_radial'] = min(df['oba_radial'])
            features['mean_oba_radial'] = mean(df['oba_radial'])
            features['median_oba_radial'] = median(df['oba_radial'])
            features['mode_oba_radial'] = mode(df['oba_radial'])
            features['skew_oba_radial'] = skew(df['oba_radial'])
            features['std_oba_radial'] = std(df['oba_radial'])

            features['max_oba_tangencial'] = max(df['oba_tangencial'])
            features['min_oba_tangencial'] = min(df['oba_tangencial'])
            features['mean_oba_tangencial'] = mean(df['oba_tangencial'])
            features['median_oba_tangencial'] = median(df['oba_tangencial'])
            features['mode_oba_tangencial'] = mode(df['oba_tangencial'])
            features['skew_oba_tangencial'] = skew(df['oba_tangencial'])
            features['std_oba_tangencial'] = std(df['oba_tangencial'])

            features['max_tachometer'] = max(df['tachometer'])
            features['min_tachometer'] = min(df['tachometer'])
            features['mean_tachometer'] = mean(df['tachometer'])
            features['median_tachometer'] = median(df['tachometer'])
            features['mode_tachometer'] = mode(df['tachometer'])
            features['skew_tachometer'] = skew(df['tachometer'])
            features['std_tachometer'] = std(df['tachometer'])

            features['max_microphone'] = max(df['microphone'])
            features['min_microphone'] = min(df['microphone'])
            features['mean_microphone'] = mean(df['microphone'])
            features['median_microphone'] = median(df['microphone'])
            features['mode_microphone'] = mode(df['microphone'])
            features['skew_microphone'] = skew(df['microphone'])
            features['std_microphone'] = std(df['microphone'])

            feat = pd.DataFrame.from_dict(features, orient='index').transpose() 
            
            result = pd.concat([result, feat], ignore_index = True)

        result.to_csv(path + '.csv', index=False, sep=';')

def main():

    paths = [
        './data/horizontal-misalignment/0.5mm',
        './data/horizontal-misalignment/1.0mm',
        './data/horizontal-misalignment/1.5mm',
        './data/horizontal-misalignment/2.0mm',
        './data/imbalance/6g',
        './data/imbalance/10g',
        './data/imbalance/15g',
        './data/imbalance/20g',
        './data/imbalance/25g',
        './data/imbalance/30g',
        './data/imbalance/35g',
        './data/normal',
        './data/overhang/ball_fault/0g',
        './data/overhang/ball_fault/6g',
        './data/overhang/ball_fault/20g',
        './data/overhang/ball_fault/35g',
        './data/overhang/cage_fault/0g',
        './data/overhang/cage_fault/6g',
        './data/overhang/cage_fault/20g',
        './data/overhang/cage_fault/35g',
        './data/overhang/outer_race/0g',
        './data/overhang/outer_race/6g',
        './data/overhang/outer_race/20g',
        './data/overhang/outer_race/35g',
        './data/underhang/ball_fault/0g',
        './data/underhang/ball_fault/6g',
        './data/underhang/ball_fault/20g',
        './data/underhang/ball_fault/35g',
        './data/underhang/cage_fault/0g',
        './data/underhang/cage_fault/6g',
        './data/underhang/cage_fault/20g',
        './data/underhang/cage_fault/35g',
        './data/underhang/outer_race/0g',
        './data/underhang/outer_race/6g',
        './data/underhang/outer_race/20g',
        './data/underhang/outer_race/35g',
        './data/vertical-misalignment/0.51mm',
        './data/vertical-misalignment/0.63mm',
        './data/vertical-misalignment/1.27mm',
        './data/vertical-misalignment/1.40mm',
        './data/vertical-misalignment/1.78mm',
        './data/vertical-misalignment/1.90mm'
    ]

    generate_features(paths)

if __name__== "__main__":
    main()