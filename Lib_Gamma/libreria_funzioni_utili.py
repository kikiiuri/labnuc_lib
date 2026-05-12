'''LIBRERIA CON FUNZIONI UTILI UTILI PER SPETTROSCOPIA GAMMA :) '''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc
from scipy.signal import find_peaks
from scipy.stats import chi2
from scipy.stats import f  # per fare test-f di Fisher
from iminuit import Minuit
from iminuit.cost import LeastSquares


# ---------------------------------------------------------
# ISTOGRAMMA INDICE DEL BIN - CONTEGGI (utilizzabile in generale)
# ---------------------------------------------------------
def istogramma(data, channels, title='Istogramma dati', x_label='Canale', y_label='Conteggi'):
  fig, ax = plt.subplots (figsize=(14, 8))
  ax.set_title (title, size=14)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.fill_between(channels, data, step='mid', color='purple', alpha=1)
  ax.step(channels, data, where='mid', color='purple', linewidth=0.5, label='Conteggi')
  ax.grid(True)


# ---------------------------------------------------------
# AUTO-RICERCA DEGLI n PICCHI MAGGIORI -> restituisce il np.array con i numeri dei canali in cui si trovano gli n picchi maggiori (in ordine crescente)
# (utilizzabile in generale in un istogramma "indice del bin (canale) - conteggi")
# ---------------------------------------------------------
def auto_ricerca_n_picchi(data, n, n_first_channels_to_ignore, prominence, distance, verbose=True, max_channel=None):

  # Verbose = True: mostra le scritte di output ("verboso, dettagliato, prolisso")
  # Ignora tutti i canali successivi a "max_channel" (se specificato)

  n = int(n) # numero picchi da cercare

  ''' Cerchiamo i picchi. Ignoriamo i primi "n_first_channels_to_ignore" canali (spesso pieni di rumore elettronico, es. 200 canali).
    Impostiamo una 'prominence' minima (es. 200 conteggi) e una distanza minima tra i picchi (es. 500 canali).'''
  peaks, properties = find_peaks(data[n_first_channels_to_ignore : max_channel], prominence=prominence, distance=distance) 
  # peaks = array con canali dei picchi trovati (ma canali diminuiti di "n_first_channels_to_ignore"!)
  # properties = dizionario con info sui picchi, tra cui: properties['prominences'] → quanto ogni picco è “importante”

  # Dobbiamo ri-aggiungere i "n_first_channels_to_ignore" canali di offset che avevamo ignorato, per correggere il numero dei canali
  peaks = peaks + n_first_channels_to_ignore 

  # Ordiniamo gli indici dell'array "peaks" in ordine crescente per prominenza e selezioniamo gli ultimi n (i maggiori -> per es. per n=2 i due veri fotopicchi del Na-22!)
  best_peaks_idx = np.argsort(properties['prominences'])[-n:] # indici in "peaks" degli n picchi maggiori
  # Ora prendiamo gli n canali associati agli n picchi maggiori trovati e ordiniamoli in ordine crescente (per canale)
  best_peaks_ch = np.sort(peaks[best_peaks_idx]) # ottengo così un array con gli n canali, ordinati in ordine crescente

  if verbose:
    print(" RICERCA AUTOMATICA DEI PICCHI COMPLETATA")
    print("="*50)
    for i, elem in enumerate(best_peaks_ch):
      print(f"Trovato Picco {i+1} a canale: {elem} (Altezza: {data[elem]:.0f})")
    print("="*50 + "\n")

  return best_peaks_ch # restituisce il np.array con i numeri dei canali dei picchi maggiori (in ordine crescente)


#-------------------------------------------------------
# PLOT DEL FIT TOTALE E ANALISI DEI RESIDUI:
# utilizzabile in generale per plottare i due axes: plot fit + plot residui)
# (per i residui tiene in conto ANCHE l'errore sull'asse x, calcolando la derivata di "model" in modo numerico!!!)
#-------------------------------------------------------
def plot_fit_with_residuals(ax_main, ax_res, model, x, y, x_err, y_err, m_fit, title, x_label='Canale', y_label='Conteggi'):

    # 1. PLOT PRINCIPALE (In alto)
    #-------------------------------------------------------
    ax_main.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='k.', label='Dati', markersize=5, ecolor='lightgray')

    # Fit Totale
    x_dense = np.linspace(np.min(x), np.max(x), 500) # Valori densi per una curva fluida
    y_fit_dense = model(x_dense, *m_fit.values)
    ax_main.plot(x_dense, y_fit_dense, 'r-', linewidth=2, label='Fit Totale')

    ax_main.set_title(f"{title}")
    ax_main.set_ylabel(y_label)
    ax_main.legend()
    ax_main.grid(True, alpha=0.4)

    # Nascondo le etichette dell'asse X sul grafico principale per pulizia
    plt.setp(ax_main.get_xticklabels(), visible=False)


    # 2. PLOT DEI RESIDUI NORMALIZZATI (In basso)
    #-------------------------------------------------------
    # Calcolo i valori teorici del fit ESATTAMENTE sui canali sperimentali
    y_fit_points = model(x, *m_fit.values)

    # --- Calcolo dell'Errore Efficace (Effective Variance) ---
    # np.any() è False se x_err è None, 0, o un array composto solo da zeri
    if x_err is not None and np.any(x_err):
        # Calcolo numerico della derivata prima df/dx tramite differenza centrale
        h = 1e-5  # Passo piccolo per la derivata
        df_dx = (model(x + h, *m_fit.values) - model(x - h, *m_fit.values)) / (2 * h)
        
        # Propagazione dell'errore: sigma_eff = sqrt(sigma_y^2 + (df/dx * sigma_x)^2)
        # (np.atleast_1d garantisce che funzioni anche se y_err è uno scalare)
        sigma_eff = np.sqrt(np.atleast_1d(y_err)**2 + (df_dx * np.atleast_1d(x_err))**2)
    else:
        sigma_eff = np.atleast_1d(y_err)

    # Calcolo i residui normalizzati con la varianza efficace
    residuals = (y - y_fit_points) / sigma_eff

    # --- Calcolo media e deviazione standard con errori associati (mi aspetto res_mean=0, res_std=1 ---
    res_mean = np.mean(residuals)
    res_std = np.std(residuals, ddof=1) 
    # ddof=1 (Delta Degrees of Freedom -> correzione dei gradi di libertà) per la dev. standard campionaria (correzione di Bessel)
    # ddof=0 (di default) per la dev. standard della popolazione (senza correzione di Bessel)
    errore_res_mean = res_std / np.sqrt(len(residuals))
    errore_res_std = res_std / np.sqrt(2 * (len(residuals) - 1)) # sotto l'ipotesi di distribuzione normale

    # Traccio i residui con le barre di errore unitarie (poiché sono normalizzati)
    ax_res.errorbar(x, residuals, yerr=1.0, fmt='k.', markersize=4, ecolor='lightgray')
    ax_res.axhline(0, color='r', linestyle='--', linewidth=1.5) # Linea dello zero

    # Traccio le bande a +/- 2 sigma e +/- 3 sigma per riferimento
    ax_res.axhspan(-2, 2, color='green', alpha=0.1, label=r'$\pm 2\sigma$')
    ax_res.axhspan(-3, 3, color='yellow', alpha=0.1, label=r'$\pm 3\sigma$')

    ax_res.set_xlabel(x_label)
    ax_res.set_ylabel(r'Residui ($\sigma$)')
    ax_res.grid(True, alpha=0.4)

    # Imposto limiti sull'asse Y fissi in base al residuo maggiore
    max_res = max(abs(residuals).max(), 4)
    ax_res.set_ylim(-max_res*1.1, max_res*1.1)

    # Aggiungo un box di testo sul grafico con i risultati
    # Posizionato in alto a sinistra (x=0.02, y=0.95 nelle coordinate degli assi)
    textstr = f'$\\mu = {res_mean:.2f}$ ± ${errore_res_mean:.2f}$\n$\\sigma = {res_std:.2f}$ ± ${errore_res_std:.2f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax_res.text(0.02, 0.90, textstr, transform=ax_res.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)


# =========================================================
# TEST F DI FISHER PER MODELLI ANNIDATI
# =========================================================
'''Il test-F è lo strumento statistico d'elezione quando fai un fit ai minimi quadrati (o calcoli il chi^2) 
e vuoi decidere se la complessità aggiuntiva di un modello è giustificata dai dati.
Consideriamo un modello semplice (Modello 1) e uno complesso (Modello 2) che contiene tutti i parametri del 
Modello 1 più alcuni parametri extra (modelli "annidati").
Poiché il Modello 2 ha più "gradi di libertà" per adattarsi ai dati, il suo errore residuo (il suo chi^2) 
sarà sempre minore o uguale a quello del Modello 1.
La domanda a cui risponde il test di Fisher è: questa riduzione del chi^2 è sufficientemente grande da 
giustificare l'uso di parametri in più, o è solo frutto del caso (cioè il modello sta fittando il rumore)?
Le due ipotesi del test sono:
- Ipotesi Nulla (H_0): Il modello semplice è quello corretto. I parametri extra del modello complesso in 
  realtà valgono zero (o non portano un miglioramento reale).
- Ipotesi Alternativa (H_1): Il modello complesso offre un miglioramento statisticamente significativo 
  ed è da preferire.'''

def confronta_modelli_ftest(m_semplice, m_complesso, alpha=0.05, nome_picco="Picco"):
    """
    Esegue un F-test tra due modelli annidati fittati con Minuit.
    m_semplice: oggetto Minuit del fit con meno parametri (es. Costante)
    m_complesso: oggetto Minuit del fit con più parametri (es. Retta)
    alpha: livello di significatività (default 5% = 0.05)
    """
    chi2_semplice = m_semplice.fval
    ndof_semplice = m_semplice.ndof
    
    chi2_complesso = m_complesso.fval
    ndof_complesso = m_complesso.ndof
    
    # Calcolo differenze
    delta_chi2 = chi2_semplice - chi2_complesso
    delta_ndof = ndof_semplice - ndof_complesso
    
    print(f"\n--- TEST F: {nome_picco} ---")
    
    # Se per qualche strano motivo il modello complesso ha un chi2 peggiore (convergenza fallita)
    if delta_chi2 <= 0:
         print("Il modello complesso ha fallito la convergenza o non ha migliorato il fit.")
         print("=> VINCITORE: Modello Semplice (Costante)")
         return m_semplice
         
    # Formula della statistica F
    f_stat = (delta_chi2 / delta_ndof) / (chi2_complesso / ndof_complesso)
    
    # Calcolo del p-value (survival function = 1 - cdf)
    p_value = f.sf(f_stat, delta_ndof, ndof_complesso)
    
    print(f"Chi2/ndof Semplice: {chi2_semplice:.2f}/{ndof_semplice} = {chi2_semplice/ndof_semplice:.2f}")
    print(f"Chi2/ndof Complesso: {chi2_complesso:.2f}/{ndof_complesso} = {chi2_complesso/ndof_complesso:.2f}")
    print(f"F-statistic: {f_stat:.2f} | p-value: {p_value:.4e}")
    
    if p_value < alpha:
        print(f"Esito: Il parametro extra è SIGNIFICATIVO (p < {alpha}).")
        print("=> VINCITORE: Modello più Complesso")
        return m_complesso
    else:
        print(f"Esito: Il parametro extra è SUPERFLUO (p >= {alpha}), stai fittando il rumore.")
        print("=> VINCITORE: Modello più Semplice")
        return m_semplice



# ====================================================================
# MODELLI E FIT PER LA CURVA DI OTTIMIZZAZIONE (RISOLUZIONE vs PARAMETRO)
# (utilizzabile per risoluzione relativa vs V_bias o shaping_time)
# ====================================================================

def modello_risoluzione(X, A, B, C):
    """ Modello fisico: R_% = sqrt(A + B/X + C*X) """
    argomento_safe = np.clip(A + (B / X) + (C * X), 1e-10, None) # per evitare valori negativi
    return np.sqrt(argomento_safe)

def derivata_modello_risoluzione(X, A, B, C):
    """ Derivata prima del modello rispetto a X (necessaria per la varianza efficace) """
    argomento_safe = np.clip(A + (B / X) + (C * X), 1e-10, None)
    return 0.5 / np.sqrt(argomento_safe) * (-B / (X**2) + C)

def studio_ottimizzazione_risoluzione(x_data, err_x_data, res_p1, err_res_p1, res_p2, err_res_p2, 
                                      picco_scelto='entrambi', nome_x='V_{bias}', unita_x='kV'):
    """
    Esegue il fit della Risoluzione Relativa tenendo conto dell'errore sull'asse X (Varianza Efficace).
    
    Parametri:
    - x_data, err_x_data: array della variabile indipendente (es. V_bias o shaping time) e suo errore
    - res_p1, err_res_p1: array della risoluzione del Picco 1 e suo errore
    - res_p2, err_res_p2: array della risoluzione del Picco 2 e suo errore
    - picco_scelto: 0 (o 'P1') per Picco 1, 1 (o 'P2') per Picco 2, 'entrambi' per plottarli entrambi.
    - nome_x: stringa col nome della variabile (es. 'V_{bias}' o 'Tempo di Formazione')
    - unita_x: stringa con l'unità di misura (es. 'kV' o '$\\mu s$')
    """

    def esegui_fit_e_trova_minimo(x_val, err_x_val, y_val, err_y_val, nome_picco):
        print(f"\n--- FIT OTTIMIZZAZIONE: {nome_picco} ---")
        
        def custom_chi2(A, B, C):
            y_model = modello_risoluzione(x_val, A, B, C)
            dy_dx = derivata_modello_risoluzione(x_val, A, B, C)
            var_eff = err_y_val**2 + (dy_dx * err_x_val)**2
            return np.sum(((y_val - y_model)**2) / var_eff)
            
        custom_chi2.errordef = Minuit.LEAST_SQUARES
        
        m = Minuit(custom_chi2, A=1.0, B=1.0, C=0.1)
        m.limits['A'] = (0, None)
        m.limits['B'] = (0, None)
        m.limits['C'] = (0, None)
        
        m.migrad()
        m.hesse()
        
        A, err_A = m.values['A'], m.errors['A']
        B, err_B = m.values['B'], m.errors['B']
        C, err_C = m.values['C'], m.errors['C']
        chi2_ndof = m.fval / (len(x_val) - m.nfit)
        
        print(f"Parametri:")
        print(f"  A = {A:.4f} ± {err_A:.4f}")
        print(f"  B = {B:.4f} ± {err_B:.4f}")
        print(f"  C = {C:.4f} ± {err_C:.4f}")
        print(f"Chi2/ndof: {chi2_ndof:.2f}")
        
        # Calcolo del Minimo e propagazione dell'errore
        if C > 1e-6: # Se C non è nullo (la curva ha una forma a U e non a L)
            x_opt = np.sqrt(B / C)
            cov_BC = m.covariance[1, 2]
            dX_dB = 0.5 / np.sqrt(B * C)
            dX_dC = -0.5 * np.sqrt(B / (C**3))
            var_x_opt = (dX_dB * err_B)**2 + (dX_dC * err_C)**2 + 2 * dX_dB * dX_dC * cov_BC
            err_x_opt = np.sqrt(var_x_opt)
            print(f"{nome_x} Ottimale = {x_opt:.2f} ± {err_x_opt:.2f} {unita_x}")
        else:
            x_opt, err_x_opt = None, None
            print(f"Parametro C compatibile con zero: nessun minimo di {nome_x} nel range trovato.")
            
        return m, x_opt, err_x_opt

    plt.figure(figsize=(10, 6))
    x_dense = np.linspace(min(x_data)*0.9, max(x_data)*1.1, 500)

    mostra_p1 = picco_scelto in [0, 'P1', 'p1', 'entrambi']
    mostra_p2 = picco_scelto in [1, 'P2', 'p2', 'entrambi']

    # --- PICCO 1 (511 keV) ---
    if mostra_p1:
        m_511, x_opt_511, _ = esegui_fit_e_trova_minimo(x_data, err_x_data, res_p1, err_res_p1, "511 keV")
        
        plt.errorbar(x_data, res_p1, xerr=err_x_data, yerr=err_res_p1, 
                     fmt='o', color='blue', ecolor='darkblue', markersize=4, elinewidth=1.5, 
                     linestyle='none', alpha=0.8, label='Dati (511 keV)')
        y_fit_511 = modello_risoluzione(x_dense, *m_511.values)
        plt.plot(x_dense, y_fit_511, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='Fit (511 keV)')
        
        if x_opt_511:
          plt.axvline(x_opt_511, color='blue', linestyle=':', alpha=0.6, 
                      label=f'${{{nome_x}}} \\ ottimale = {x_opt_511:.2f}$ {unita_x}')
            
    # --- PICCO 2 (1274.5 keV) ---
    if mostra_p2:
        m_1274, x_opt_1274, _ = esegui_fit_e_trova_minimo(x_data, err_x_data, res_p2, err_res_p2, "1274.5 keV")
        
        plt.errorbar(x_data, res_p2, xerr=err_x_data, yerr=err_res_p2, 
                     fmt='s', color='red', ecolor='darkred', markersize=4, elinewidth=1.5, 
                     linestyle='none', alpha=0.8, label='Dati (1274.5 keV)')
        y_fit_1274 = modello_risoluzione(x_dense, *m_1274.values)
        plt.plot(x_dense, y_fit_1274, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Fit (1274.5 keV)')
        
        if x_opt_1274:
          plt.axvline(x_opt_1274, color='red', linestyle=':', alpha=0.6, 
                      label=f'${{{nome_x}}} \\ ottimale = {x_opt_1274:.2f}$ {unita_x}')
    
    plt.title(f'Risoluzione Relativa Percentuale vs ${nome_x}$', fontsize=14)
    plt.xlabel(f'${nome_x}$ [{unita_x}]', fontsize=12)
    plt.ylabel('Risoluzione Relativa [%]', fontsize=12)

    plt.grid(True, which='major', linestyle='-', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.minorticks_on()

    plt.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()



# ====================================================================
# CALIBRAZIONE ENERGETICA
# ====================================================================

# --------------------------------------
# Modelli matematici per la calibrazione:
# --------------------------------------

def modello_calibrazione(x, q, m):
    """
    Modello lineare per la calibrazione: E = m * Ch + q.
    Può essere facilmente sostituito con un modello quadratico se necessario.
    """
    return m * x + q

def derivata_calibrazione(x, q, m):
    """
    Derivata prima del modello rispetto alla variabile indipendente x (Ch).
    Serve per proiettare l'errore dell'asse X sull'asse Y (Metodo della Varianza Effettiva).
    Per una retta, la derivata è semplicemente la pendenza 'm'.
    """
    # Usiamo np.full_like in modo che restituisca sempre un array della stessa
    # dimensione di x, prevenendo errori matematici se in futuro si usa una
    # derivata dipendente da x (es. per una parabola: 2*a*x + b).
    return np.full_like(x, m, dtype=float)

# -------------------------------------------------------
# FUNZIONE DI FIT, tenendo in considerazione sia l'errore 
# sull'asse x che sull'asse y (Varianza Effettiva)
# -------------------------------------------------------

def calibrazione_energetica(E_nom, err_E_nom, mu_ch, err_mu_ch, mostra_plot=True):
    """
    Esegue la calibrazione energetica tenendo conto sia degli errori 
    sull'asse X (canali) che sull'asse Y (energie).
    
    Parametri:
    - E_nom: array-like, energie nominali dei picchi (asse Y)
    - err_E_nom: array-like, errori sulle energie nominali (asse Y)
    - mu_ch: array-like, centroidi fittati in canali (asse X)
    - err_mu_ch: array-like, errori sui centroidi (asse X)
    - mostra_plot: bool, se True genera il grafico
    """
    # 1. Conversione e validazione dei dati
    E_nom = np.asarray(E_nom, dtype=float)
    err_E_nom = np.asarray(err_E_nom, dtype=float)
    mu_ch = np.asarray(mu_ch, dtype=float)
    err_mu_ch = np.asarray(err_mu_ch, dtype=float)
    
    # 2. Stima iniziale dei parametri (Guess)
    m_guess = (E_nom[-1] - E_nom[0]) / (mu_ch[-1] - mu_ch[0])
    q_guess = E_nom[0] - m_guess * mu_ch[0]
    
    # 3. Funzione di costo personalizzata (Varianza Effettiva)
    def cost_function_xy_errors(q, m):
        # Valore atteso dal modello (chiamata alla funzione esterna)
        E_fit = modello_calibrazione(mu_ch, q, m)
        
        # Derivata del modello rispetto a X valutata nei centroidi
        dE_dx = derivata_calibrazione(mu_ch, q, m)
        
        # Varianza effettiva: sigma_Y^2 + (dE/dx)^2 * sigma_X^2
        var_eff = err_E_nom**2 + (dE_dx**2) * (err_mu_ch**2)
        
        # Chi-quadro da minimizzare
        return np.sum(((E_nom - E_fit)**2) / var_eff)

    # 4. Inizializzazione di Minuit
    m_fit = Minuit(cost_function_xy_errors, q=q_guess, m=m_guess)
    m_fit.errordef = Minuit.LEAST_SQUARES # delta_chi2 = 1.0 per errori 1-sigma
    
    # 5. Esecuzione del Fit
    m_fit.migrad()
    m_fit.hesse() # Matrice di covarianza esatta
    
    # 6. Estrazione dei risultati
    q_opt = m_fit.values['q']
    m_opt = m_fit.values['m']
    err_q = m_fit.errors['q']
    err_m = m_fit.errors['m']
    cov_matrix = np.array(m_fit.covariance)
    
    chi2_val = m_fit.fval
    ndof = len(E_nom) - m_fit.nfit
    p_value = chi2.sf(chi2_val, ndof)
    
    # 7. Stampa a schermo
    print(f"\n{'='*50}")
    print(" RISULTATI CALIBRAZIONE ENERGETICA")
    print(f"{'='*50}")
    print(f"Modello: E(Ch) = m * Ch + q")
    print(f"Gain (m)     : {m_opt:.8f} ± {err_m:.8f} keV/ch")
    print(f"Offset (q)   : {q_opt:.8f} ± {err_q:.8f} keV")
    print(f"Chi2 / ndof  : {chi2_val:.2f} / {ndof} = {(chi2_val/ndof):.2f}")
    print(f"p-value (%)  : {p_value*100:.4f} %")
    print(f"{'='*50}\n")
    
    # 8. Plot con residui (se richiesto)
    if mostra_plot:
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        
        ax_main = fig.add_subplot(gs[0, 0])
        ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)
        
        # Sfruttiamo la funzione di libreria esistente!
        # Chiamata esplicita con keyword arguments per massima sicurezza
        plot_fit_with_residuals(
            ax_main=ax_main, 
            ax_res=ax_res, 
            model=modello_calibrazione, 
            x=mu_ch, 
            y=E_nom, 
            x_err=err_mu_ch, 
            y_err=err_E_nom, 
            m_fit=m_fit, 
            title="Retta di Calibrazione",
            x_label='Centroide (Canali)',
            y_label='Energia [keV]'
        )
        
        plt.show()

    # Ritorna un dizionario con i risultati
    return {
        'q': q_opt,
        'm': m_opt,
        'err_q': err_q,
        'err_m': err_m,
        'cov_matrix': cov_matrix,
        'chi2': chi2_val,
        'ndof': ndof,
        'p_value': p_value
    }
