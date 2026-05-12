'''LIBRERIA CON MODELLO PDF HYPERMET STANDARD, FIT E ANALISI PER SPETTROSCOPIA GAMMA''' 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc
from scipy.signal import find_peaks
from scipy.stats import chi2
from iminuit import Minuit
from iminuit.cost import LeastSquares
import Lib_Gamma.libreria_funzioni_utili as lfu  # libreria creata, nella cartella "Librerie"

#_______________________________________________________________________________

# ---------------------------------------------------------
# MODELLO PDF: MODELLO HYPERMET STANDARD
# ---------------------------------------------------------
def Hypermet_standard_model(x, a, mu, sigma, tail_amp, tail_beta, step_amp, b0, b1):
    """
    Modello Hypermet per picchi gamma in rivelatori HPGe:
    Gaussiana + Coda Esponenziale a sinistra + Gradino multi-Compton + Fondo Lineare
    """
    # 1. COMPONENTE GAUSSIANA
    # a = ampiezza massima della gaussiana
    # mu = media -> centroide del picco (proporzionale all'energia nominale del gamma)
    # sigma = dev standard
    gauss = a * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # 2. CODA ESPONENZIALE a bassa energia (Incomplete Charge Collection)
    # tail_amp = ampiezza della coda
    # tail_beta = rappresenta la pendenza (decay constant) dell'esponenziale verso le basse energie
    erfc_tail_arg = (x - mu) / (np.sqrt(2) * sigma) + (tail_beta * sigma) / np.sqrt(2)
    exp_arg = np.clip(tail_beta * (x - mu), -100, 100) 
    # np.clip evita errori di overflow nell'esponenziale durante i tentativi di Minuit (limita l'argomento tra -100 e +100)
    tail = tail_amp * np.exp(exp_arg) * erfc(erfc_tail_arg) / 2.0

    # 3. GRADINO MULTI-COMPTON 
    # step_amp = altezza del gradino (la differenza di conteggi tra il fondo a sinistra e il fondo a destra del picco)
    erfc_step_arg = (x - mu) / (np.sqrt(2) * sigma)
    step = step_amp * erfc(erfc_step_arg) / 2.0

    # 4. FONDO LINEARE
    # b0 = altezza del fondo sotto il picco
    # b1 = pendenza locale
    linear_bg = b0 + b1 * (x - mu)

    return gauss + tail + step + linear_bg


# ---------------------------------------------------------
# FUNZIONE DI FIT HYPERMET STANDARD (automatizzata a due passaggi)
# ---------------------------------------------------------
def fit_Hypermet_standard(roi_x, roi_y, err_y, guess_params):

    # Inizializziamo il Chi2
    cost = LeastSquares(roi_x, roi_y, err_y, Hypermet_standard_model)

    # WORKAROUND: Passiamo i valori in modo posizionale, nell'esatto
    # ordine in cui appaiono nella funzione Hypermet_standard_model!
    m = Minuit(cost,
               guess_params['a'],
               guess_params['mu'],
               guess_params['sigma'],
               guess_params['tail_amp'],
               guess_params['tail_beta'],
               guess_params['step_amp'],
               guess_params['b0'],
               guess_params['b1'])

    '''# Diciamo esplicitamente a Minuit che stiamo facendo un Least Squares
    (buona pratica anche se non è necessaria in questo caso, Minuit già lo sa, dato che abbiamo 
       scritto cost = LeastSquares() -> ha senso invece quando si costruisce la funzione cost, cioé chi^2, a mano)
    m.errordef = Minuit.LEAST_SQUARES '''
    
    # Impostiamo limiti fisici sensati (usando i nomi delle variabili)
    m.limits['a'] = (0, None)
    m.limits['sigma'] = (0.1, None) # ha senso 0.1 come limite inferiore perché siamo in canali ancora
    m.limits['tail_amp'] = (0, None)
    m.limits['tail_beta'] = (0, None)
    m.limits['step_amp'] = (0, None)

    # --- PASSAGGIO 1: Fit Semplificato ---
    m.fixed['tail_amp'] = True
    m.fixed['tail_beta'] = True
    m.fixed['step_amp'] = True

    m.migrad()

    # --- PASSAGGIO 2: Fit Completo ---
    m.fixed['tail_amp'] = False
    m.fixed['tail_beta'] = False
    m.fixed['step_amp'] = False

    m.migrad()
    m.hesse()

    return m


# =========================================================
# SODIO 22 con HPGe: ANALISI CON MODELLO HYPERMET STANDARD
# =========================================================
def hypermet_standard_analisi_sodio22(data, channels, window=55, n_first_channels_to_ignore=200, prominence=200, distance=500):

  #-------------------------------------------------------
  # 1. Plot dell'istogramma
  #-------------------------------------------------------
  lfu.istogramma(data, channels)

  #-------------------------------------------------------
  # 2. Auto-ricerca dei due picchi e definizione ROI:
  #-------------------------------------------------------
  picchi_ch = lfu.auto_ricerca_n_picchi(data, 2, n_first_channels_to_ignore, prominence, distance)
  picco_1_ch = picchi_ch[0]
  picco_2_ch = picchi_ch[1]

  # Definizione ROI ("Region of interest" -> regione di interesse)
  # window = semi-larghezza della ROI (es. 55 canali a destra e sinistra)

  roi_511_x = channels[picco_1_ch - window : picco_1_ch + window]
  roi_511_y = data[picco_1_ch - window : picco_1_ch + window]
  err_roi_511_y = np.sqrt(roi_511_y)  # Calcolo incertezze (statistica di Poisson)
  err_roi_511_y[err_roi_511_y == 0] = 1.0

  roi_1274_x = channels[picco_2_ch - window : picco_2_ch + window]
  roi_1274_y = data[picco_2_ch - window : picco_2_ch + window]
  err_roi_1274_y = np.sqrt(roi_1274_y)  # Calcolo incertezze (statistica di Poisson)
  err_roi_1274_y[err_roi_1274_y == 0] = 1.0

  #-------------------------------------------------------
  # 3. ESECUZIONE DEI FIT SUI DUE PICCHI
  #-------------------------------------------------------
  # --- Picco a 511 keV ---
  #------------------------
  # Parametri Iniziali (guess)
  guess_511 = {
      'a': data[picco_1_ch], # L'altezza reale trovata! 
      'mu': picco_1_ch, # Il canale esatto trovato!
      'sigma': 10,
      'tail_amp': 500, 
      'tail_beta': 0.1,
      'step_amp': 200, 
      'b0': data[picco_1_ch - window], # Stima del fondo dal bordo della ROI
      'b1': 0
  }
  m_511 = fit_Hypermet_standard(roi_511_x, roi_511_y, err_roi_511_y, guess_511)
  
  # Estrazione Risultati 511
  c511, c511_err = m_511.values['mu'], m_511.errors['mu']
  s511, s511_err = m_511.values['sigma'], m_511.errors['sigma']
  fwhm_511_ch = 2 * np.sqrt(2 * np.log(2)) * s511
  fwhm_511_ch_err = 2 * np.sqrt(2 * np.log(2)) * s511_err

  chi2_511_observed = m_511.fval  # chi-quadrato osservato con i parametri ottimizzati
  p_value_511 = chi2.sf(chi2_511_observed, m_511.ndof)  # p-value (calcolato con la funzione di distribuzione cumulativa inversa (sopravvivenza) del chi-quadrato)
  p_value_511_percentage = p_value_511 * 100  # p-value in percentuale

  # --- Picco a 1274.5 keV ---
  #---------------------------
  # Parametri Iniziali (guess)
  guess_1274 = {
      'a': data[picco_2_ch], # L'altezza reale trovata! 
      'mu': picco_2_ch, # Il canale esatto trovato! 
      'sigma': 15,
      'tail_amp': 50, 
      'tail_beta': 0.05,
      'step_amp': 50, 
      'b0': data[picco_2_ch - window], # Stima del fondo dal bordo della ROI, 
      'b1': 0
  }
  m_1274 = fit_Hypermet_standard(roi_1274_x, roi_1274_y, err_roi_1274_y, guess_1274)

  # Estrazione Risultati 1274
  c1274, c1274_err = m_1274.values['mu'], m_1274.errors['mu']
  s1274, s1274_err = m_1274.values['sigma'], m_1274.errors['sigma']
  fwhm_1274_ch = 2 * np.sqrt(2 * np.log(2)) * s1274
  fwhm_1274_ch_err = 2 * np.sqrt(2 * np.log(2)) * s1274_err

  chi2_1274_observed = m_1274.fval  # chi-quadrato osservato con i parametri ottimizzati
  p_value_1274 = chi2.sf(chi2_1274_observed, m_1274.ndof)  # p-value (calcolato con la funzione di distribuzione cumulativa inversa (sopravvivenza) del chi-quadrato)
  p_value_1274_percentage = p_value_1274 * 100  # p-value in percentuale

  '''
  #-------------------------------------------------------
  # 4. CALIBRAZIONE (è lineare, solo approssimativa quindi, con sole due coppie energia-canale)
  #-------------------------------------------------------
  E1, E2 = 511.0, 1274.5

  # Chiamo la funzione passando i dati specifici di questo esperimento
  risultati_calib = lfu.calibra_spettro(
      E1, E2, 
      c511, c1274, 
      c511_err, c1274_err, 
      channels, data, 
      mostra_plot=True
  )
  # Estrazione dei valori dal dizionario
  m = risultati_calib['m']
  m_err = risultati_calib['m_err']
  q = risultati_calib['q']
  q_err = risultati_calib['q_err']
  cov_mq = risultati_calib['cov_mq']

  # Calcolo FWHM in keV
  fwhm_511_keV = fwhm_511_ch * m
  fwhm_1274_keV = fwhm_1274_ch * m

  fwhm_511_keV_err = fwhm_511_keV * np.sqrt((fwhm_511_ch_err/fwhm_511_ch)**2 + (m_err/m)**2)
  fwhm_1274_keV_err = fwhm_1274_keV * np.sqrt((fwhm_1274_ch_err/fwhm_1274_ch)**2 + (m_err/m)**2)
  '''
  #-------------------------------------------------------
  # 5. STAMPA DEI RISULTATI
  #-------------------------------------------------------
  print("="*60)
  print(" RISULTATI ANALISI SODIO22 CON HPGe (usando modello pdf HYPERMET STANDARD):")
  print("="*60)
  print(f"--- Picco 511 keV ---")
  print(f"Centroide : {c511:.2f} ± {c511_err:.2f} ch")
  print(f"FWHM (gaussiana) : {fwhm_511_ch:.10f} ± {fwhm_511_ch_err:.10f} ch")
  print(f"Chi2/ndof : {m_511.fval:.1f} / {m_511.ndof} = {m_511.fval/m_511.ndof:.2f}")
  print(f"p-value : {p_value_511_percentage:.10f} %")

  print(f"\n--- Picco 1274.5 keV ---")
  print(f"Centroide : {c1274:.2f} ± {c1274_err:.2f} ch")
  print(f"FWHM (gaussiana): {fwhm_1274_ch:.10f} ± {fwhm_1274_ch_err:.10f} ch")
  print(f"Chi2/ndof : {m_1274.fval:.1f} / {m_1274.ndof} = {m_1274.fval/m_1274.ndof:.2f}")
  print(f"p-value : {p_value_1274_percentage:.10f} %")
  print("="*60 + "\n")

  '''
  print("\n" + "="*60)
  print(" CALIBRAZIONE E RISOLUZIONE")
  print("="*60)
  print(f"energia = m * canale + q")
  print(f"Guadagno (m) : ({m:.5f} ± {m_err:.5f}) keV/ch")
  print(f"Zero (q)     : ({q:.2f} ± {q_err:.2f}) keV")
  print(f"Covarianza tra m e q : {cov_mq:.2f}")
  print("-" * 60)
  print(f"Risoluzione FWHM (511 keV)  : ({fwhm_511_keV:.10f} ± {fwhm_511_keV_err:.10f}) keV")
  print(f"Risoluzione FWHM (1274 keV) : ({fwhm_1274_keV:.10f} ± {fwhm_1274_keV_err:.10f}) keV")
  print("="*60 + "\n")
  '''

  #-------------------------------------------------------
  # 6. VISUALIZZAZIONE GRAFICA (CON ANALISI DEI RESIDUI)
  #-------------------------------------------------------
  # Impostazione della figura con GridSpec
  fig = plt.figure(figsize=(16, 8), constrained_layout=True)
  # Creo una griglia 2x2. Le righe hanno altezze diverse (3:1)
  gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.1)
  # gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.1, figure=fig) in alternativa

  # Assegno i subplot per il picco a 511 keV (condividendo l'asse X)
  ax1_main = fig.add_subplot(gs[0, 0])
  ax1_res = fig.add_subplot(gs[1, 0], sharex=ax1_main)

  # Assegno i subplot per il picco a 1274 keV
  ax2_main = fig.add_subplot(gs[0, 1])
  ax2_res = fig.add_subplot(gs[1, 1], sharex=ax2_main)

  # Chiamo la funzione di plottaggio del fit e dei residui
  lfu.plot_fit_with_residuals(ax1_main, ax1_res, Hypermet_standard_model, roi_511_x, roi_511_y, 0, err_roi_511_y, m_511, 'Annichilazione 511 keV')
  lfu.plot_fit_with_residuals(ax2_main, ax2_res, Hypermet_standard_model, roi_1274_x, roi_1274_y, 0, err_roi_1274_y, m_1274, 'Decadimento 1274.5 keV')

  # Plotto la componente gaussiana e il background per i due picchi
  m_list = [m_511, m_1274]
  ax_main_list = [ax1_main, ax2_main]
  roi_511_x_dense = np.linspace(roi_511_x[0], roi_511_x[-1], 500) # Valori densi per una curva fluida
  roi_1274_x_dense = np.linspace(roi_1274_x[0], roi_1274_x[-1], 500) # Valori densi per una curva fluida
  roi_x_dense_list = [roi_511_x_dense, roi_1274_x_dense]
  for i in range(0, 2):
      m_fit = m_list[i]
      v = m_fit.values
      # Plot gaussiana
      gauss_only = v[0] * np.exp(-(roi_x_dense_list[i] - v[1])**2 / (2 * v[2]**2)) + v[6]
      ax_main_list[i].plot(roi_x_dense_list[i], gauss_only, 'b--', alpha=0.6, label='Gaussiana')
      # Plot background
      bg_components = Hypermet_standard_model(roi_x_dense_list[i], 0, v[1], v[2], v[3], v[4], v[5], v[6], v[7])
      ax_main_list[i].plot(roi_x_dense_list[i], bg_components, 'g-', label='Fondo + Coda + Step')
      ax_main_list[i].legend()

  plt.show()

#_______________________________________________________________________________