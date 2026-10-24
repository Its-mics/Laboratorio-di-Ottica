# %%
import ROOT
import numpy as np
import pandas as pd
 
ROOT.gStyle.SetOptFit(1111)    # aggiunge il label con chi2/ndf e probabilita e parametri se metti 0000 toglie i dati
ROOT.gStyle.SetFitFormat(
    "4.2e"
)  # con 2 cifre decimali


#fit 
def Sellmeier(x, par):
    l = x[0]
    A = par[0]
    B = par[1]
    return np.sqrt(1 + ( (A*l**2)/(l**2 - B**2) ))

def modellopy(l, A, B):         
    return np.sqrt(1 + (A*l**2)/(l**2 - B**2))


def Sell_deriv(x, par):
    l = x
    dl = 1e-6
    return ((modellopy(l+dl/2, par[0], par[1])-modellopy(l-dl/2, par[0], par[1]))/dl)
    


#leggere file excel 
xlsx ="3.xlsx"
df = pd.read_excel(xlsx, sheet_name= "Dati")
df = df.dropna(subset=["lambda", "n'", "err_lambda_max", "sigma_n"])
df = df.to_numpy(dtype = 'float64')

x = df[:, 2]     
y = df[:, 11] 

err_x = df[:, 3]/np.sqrt(3)
err_y = df[:, 13]


index = [0,1]

x = np.delete(x, index)
y = np.delete(y, index)
err_x = np.delete(err_x, index)
err_y = np.delete(err_y, index)


#minimizzazzione con ROOT
SellFit = ROOT.TF1('model', Sellmeier, min(x), max(x), 2)
SellFit.SetParameters(1.0, 100.0)     
SellFit.SetParNames("A", "B")

gr1 = ROOT.TGraphErrors(len(x), x, y, err_x, err_y)
gr1.SetTitle("Dati sperimentali e fit; #lambda [nm]; n")
gr1.SetMarkerStyle(20)
gr1.SetMarkerSize(0.7)                    
gr1.SetMarkerColor(ROOT.kAzure + 2)
gr1.SetLineColor(ROOT.kAzure + 3)        
gr1.SetLineWidth(1)


fit_res = gr1.Fit(SellFit, "SR") # SR significa che voglio fare il fit senza disegnare la funzione e senza stampare a video i risultati del fit 

fitStatus = fit_res.Status()
print(f"Fit status: {fitStatus}")

par = SellFit.GetParameters()

errors = SellFit.GetParErrors()

chi2 = fit_res.Chi2()
ndf = fit_res.Ndf()
chi2_sell, ndf_sell = chi2, ndf      

err_yeff = np.sqrt(err_y**2 + ((Sell_deriv(x, par)) * err_x)**2) #errore effettivo

print("Parametri stimati:")
print(f"A = {par[0]} ± {errors[0]}")
print(f"B = {par[1]} ± {errors[1]}")
print(f"Chi2 = {chi2}, ndf = {ndf}")

c1 = ROOT.TCanvas("c1", "Fit con ROOT", 800, 600)
c1.SetGrid()

gr1.Draw("AP")


leg1 = ROOT.TLegend(0.17, 0.80, 0.45, 0.89)
leg1.AddEntry(gr1, "Dati", "pe")
leg1.AddEntry(SellFit, "Fit", "l")
leg1.Draw()

c1.Draw()
c1.SaveAs("Immagini/fit_root.png")
ROOT.gStyle.SetOptFit(0000)    

# %%
if np.abs(par[0] - 1) < 3*errors[0]:
        print("A è compatibile con 1")
else :
        print("A NON è compatibile con 1")

print("---")

if np.abs(par[1] - 100) < 3*errors[1]:
    print("B è compatibile con 100")
else :
    print("B NON è compatibile con 100")
 
 


# %%




A_opt= 1.5
B_opt = -5e-5

A_vals = np.linspace(A_opt - 1, A_opt + 1, 500)  
B_vals = np.linspace(B_opt - 2e-3, B_opt + 2e-3, 1000)  

h2 = ROOT.TH2D(
    "chi2_surface", 
    "Scansione 3D del #chi^{2}; Parametro A; Parametro B; #chi^{2}", 
    len(A_vals), A_vals[0], A_vals[-1],
    len(B_vals), B_vals[0], B_vals[-1]
)

for i, B in enumerate(B_vals):
    w = 1 / (err_y**2 + (B*err_x)**2)
    for j, A in enumerate(A_vals):
        chi2 = np.sum(w * (y - A - B*x)**2)
        h2.SetBinContent(j + 1, i + 1, chi2)

c_scan3d = ROOT.TCanvas("c_scan3d", "Scansione 3D chi2", 900, 700)
c_scan3d.SetGrid()
c_scan3d.SetLogz()  

h2.Draw("SURF1")
ROOT.gStyle.SetOptFit(0000)  
c_scan3d.Modified()
c_scan3d.Update()
c_scan3d.Draw()

# %%
 
def lineare(x, par):
    return par[0] + par[1] * x[0]

LinFit = ROOT.TF1('lineare', lineare, min(x), max(x), 2)
LinFit.SetParameters(1.52, 0)         
LinFit.SetParNames("A", "B")
LinFit.SetLineColor(ROOT.kGreen + 2)
LinFit.SetLineStyle(2)

gr_lin = ROOT.TGraphErrors(len(x), x, y, err_x, err_y)
gr_lin.SetTitle("Fit Lineare; #lambda [nm]; n")
gr_lin.SetMarkerStyle(20)
gr_lin.SetMarkerSize(0.7)
gr_lin.SetMarkerColor(ROOT.kAzure + 2)
gr_lin.SetLineColor(ROOT.kAzure + 3)
gr_lin.SetLineWidth(1)

res_lin = gr_lin.Fit(LinFit, "SR")
print(f"Fit status: {res_lin.Status()}")

par_lin = LinFit.GetParameters()
err_lin = LinFit.GetParErrors()
chi2_lin = res_lin.Chi2()
ndf_lin = res_lin.Ndf()

print("Parametri stimati (fit lineare):")
print(f"A = {par_lin[0]} ± {err_lin[0]}")
print(f"B = {par_lin[1]} ± {err_lin[1]}")

print()
print(f"Sellmeier: chi2/ndf = {chi2_sell:.2f}/{ndf_sell} ")
print(f"Lineare:   chi2/ndf = {chi2_lin:.2f}/{ndf_lin} ")
print(f"Delta chi2 (lineare - Sellmeier) = {chi2_lin - chi2_sell:.2f}")


c_lin = ROOT.TCanvas("c_lin", "Fit lineare", 800, 600)
c_lin.SetGrid()

gr_lin.Draw("AP")
LinFit.Draw("SAME")


leg_lin = ROOT.TLegend(0.14, 0.14, 0.44, 0.24)
leg_lin.SetTextSize(0.03)  
leg_lin.AddEntry(gr_lin, "Dati", "pe")
leg_lin.AddEntry(LinFit, "Fit lineare", "l")
leg_lin.Draw()

c_lin.Modified()
c_lin.Update()
c_lin.Draw()
c_lin.SaveAs("Immagini/fit_lineare.png")

c_comp = ROOT.TCanvas("c_comp", "Confronto modelli", 800, 600)
c_comp.SetGrid()

gr_comp = gr_lin.Clone("gr_comp")      
gr_comp.SetTitle("Confronto tra i modelli; #lambda [nm]; n")
gr_comp.Draw("AP")
LinFit.Draw("SAME")
SellFit.Draw("SAME")

leg_comp = ROOT.TLegend(0.14, 0.14, 0.44, 0.28)
leg_comp.SetTextSize(0.03)  
leg_comp.AddEntry(gr_comp, "Dati", "pe")
leg_comp.AddEntry(LinFit, "Fit lineare", "l")
leg_comp.AddEntry(SellFit, "Fit Sellmeier", "l")
leg_comp.Draw()

c_comp.Modified()
c_comp.Update()
c_comp.Draw()
c_comp.SaveAs("Immagini/fit_confronto.png")




# %%

y_attesi = np.array([SellFit.Eval(x_val) for x_val in x])
residuals_norm = (y - y_attesi) / err_yeff


gaussian_fit = ROOT.TF1('gaussian', 'gaus', min(residuals_norm), max(residuals_norm), 3)
gaussian_fit.SetParameters(1, 0, 1)
gaussian_fit.SetParNames("const", "mu", "sigma")
gaussian_fit.SetLineColor(ROOT.kRed + 1)
gaussian_fit.SetLineWidth(3)

hist = ROOT.TH1F("hist", "Istogramma dei dati; x; y", 7, min(residuals_norm), max(residuals_norm))
for val in residuals_norm:
    hist.Fill(val)
    
hist.SetLineColor(ROOT.kAzure + 3)
hist.SetLineWidth(2)
hist.SetFillColorAlpha(ROOT.kAzure + 1, 0.55)   
hist.GetXaxis().SetTitle("residuo normalizzato")
hist.GetYaxis().SetTitle("conteggi")
hist.SetTitle("Istogramma residui normalizzati")

r = hist.Fit(gaussian_fit, "S")

fitStatus = r.Status()
print(f"Fit status: {fitStatus}")

parameters = gaussian_fit.GetParameters()
errors = gaussian_fit.GetParErrors()

chi2 = gaussian_fit.GetChisquare()

print("Parametri stimati:")
print(f"const = {parameters[0]} ± {errors[0]}")
print(f"#mu = {parameters[1]} ± {errors[1]}")
print(f"#sigma = {parameters[2]} ± {errors[2]}")
print(f"Chi2 = {chi2}")


c2 = ROOT.TCanvas("c2", "Istogramma dei dati", 800, 600)
c2.SetGrid()

hist.Draw()

leg2 = ROOT.TLegend(0, 0.30, 0.15, 0.39)
leg2.AddEntry(hist, "Residui", "f")
leg2.AddEntry(gaussian_fit, "Fit gaussiano", "l")
leg2.Draw()

ROOT.gStyle.SetOptFit(1111)  
c2.Draw()
c2.SaveAs("Immagini/hist.png")


# %%
if np.abs(parameters[0] - 1) < 3*errors[0]:
    print("const è compatibile con 1")
else :
    print("const NON è compatibile con 1")

print("---")

if np.abs(parameters[1]) < 3*errors[1]:
    print("mu è compatibile con 0")
else :
    print("mu NON è compatibile con 0")

print("---")
if np.abs(parameters[2] - 1) < 3*errors[2]:
    print("sigma è compatibile con 1")
else :
    print("sigma NON è compatibile con 1")

# %%
def line(x, par):
    x = x[0]
    m = par[0]
    q = par[1]
    return m * x + q

residuals = y - y_attesi

line_fit = ROOT.TF1('line', line, min(x), max(x), 2)
line_fit.SetParameters(1, 0)
line_fit.SetParNames("m", "q")
line_fit.SetLineColor(ROOT.kRed + 1)
line_fit.SetLineWidth(2)

gr2 = ROOT.TGraphErrors(len(x), x, residuals, np.zeros_like(residuals), err_yeff)
gr2.SetTitle("Residui in funzione di #lambda; #lambda [nm]; n - n_{att}")
gr2.SetMarkerStyle(20)
gr2.SetMarkerSize(0.7)
gr2.SetMarkerColor(ROOT.kAzure + 2)
gr2.SetLineColor(ROOT.kAzure + 3)

fit_res = gr2.Fit(line_fit, "S")

fitStatus = fit_res.Status()
print(f"Fit status: {fitStatus}")

parameters = line_fit.GetParameters()
errors = line_fit.GetParErrors()

chi2 = fit_res.Chi2()

print("Parametri stimati:")
print(f"m = {parameters[0]} ± {errors[0]}")
print(f"q = {parameters[1]} ± {errors[1]}")
print(f"Chi2 = {chi2}")

c3 = ROOT.TCanvas("c3", "Fit con ROOT", 800, 600)
c3.SetGrid()

gr2.Draw("AP")

leg3 = ROOT.TLegend(0.17, 0.80, 0.45, 0.89)
leg3.AddEntry(gr2, "Residui", "pe")
leg3.AddEntry(line_fit, "Fit lineare", "l")
leg3.Draw()

c3.Draw()
c3.SaveAs("Immagini/fit_line.png")

# %% [markdown]
# ## residui punto per punto
# su residuo/sigma (i punti con <--- sono oltre 2 sigma) e la colonna tacche
# quante tacche del nonio di errore su delta servono per spiegare lo scarto.
# 

# %%
alpha = np.pi/4                                

n_fit    = modellopy(x, SellFit.GetParameter(0), SellFit.GetParameter(1))
residuo  = y - n_fit
res_norm = residuo / err_yeff      # in di sigma


delta     = 2*np.arcsin(y*np.sin(alpha/2)) - alpha
dn_ddelta = np.cos((alpha + delta)/2) / (2*np.sin(alpha/2))
tacche    = np.degrees(residuo / dn_ddelta) * 60  # in tacche

tabella = pd.DataFrame({
    "lambda":        x,
    "n misurato":    y,
    "n fit":         n_fit,
    "residuo":       residuo,
    "residuo/sigma": res_norm,
    "tacche":        tacche,
    "oltre 2 sigma": abs(res_norm) > 2,
})
display(tabella.sort_values("lambda").round({"n misurato": 5, "n fit": 5, "residuo": 5, "residuo/sigma": 2, "tacche": 1}))

print(f"residui normalizzati: media = {res_norm.mean():.2f}, dev.std = {res_norm.std(ddof=1):.2f}  (attesi 0 e 1)")

# %% [markdown]
# ## fit togliendo dei punti
# guardo quanto cambiano A e B e il chi2

# %%
def fit_senza(lambda_tolte):

    tieni = []                              
    for i in range(len(x)):                   
        if x[i] not in lambda_tolte:      
            tieni.append(i)                  

    funzione = ROOT.TF1("f_tmp", Sellmeier, min(x), max(x), 2)    
    funzione.SetParameters(1.27, 100)                            
    grafico = ROOT.TGraphErrors(len(tieni), x[tieni], y[tieni], err_x[tieni], err_y[tieni])

    return grafico.Fit(funzione, "SRQ")      


casi = [[], [671, 690], [620, 656], [620, 656, 671, 690]]    

for lambda_tolte in casi:
    r = fit_senza(lambda_tolte)
    print("punti tolti (lambda):", lambda_tolte)
    print("   A    =", round(r.Parameter(0), 4), "+-", round(r.ParError(0), 4))
    print("   B    =", round(r.Parameter(1), 1), "+-", round(r.ParError(1), 1))
    print("   chi2 =", round(r.Chi2(), 1), "   ndf =", r.Ndf(), "   p =", round(r.Prob(), 2), " chi2_/ndf =", round(r.Chi2() / r.Ndf(), 2), "\n")   # ndf = punti - 2 parametri

# %% [markdown]
# ## ripetibilità theta0
# se lo zero varia molto piu' della sigma assunta, l'errore su delta e' sottostimato
# e uno scarto di 2-3 tacche è normale

# %%
theta0 = df[:, 0]                                 
tacca = 1/60                                     
sigma_lettura = tacca / np.sqrt(3)                 
dev_std = np.std(theta0, ddof=1)

print(f"theta_0 [gradi]: media = {theta0.mean():.3f}   min = {theta0.min():.3f}   max = {theta0.max():.3f}")
print(f"dev.std di theta_0       = {dev_std:.3f}")
print(f"sigma assunta (lettura)  = {sigma_lettura:.4f}      (1 tacca = {tacca:.4f})")
print(f"dev.std / sigma assunta  = {dev_std / sigma_lettura:.1f}")

# %% [markdown]
# ## filtri con altra lambda
# fit sui soli punti fino a 589 nm poi per ogni n misurato si ricava la lambda che la legge richiede

# %%
buoni = x <= 589
r = fit_sellmeier(x[buoni], y[buoni], err_x[buoni], err_y[buoni])
A_b = r.Parameter(0)
B_b = r.Parameter(1)
print(f"fit sui punti <= 589 nm: A = {A_b:.4f}   B = {B_b:.1f}")

lam_implicita = B_b * np.sqrt((y**2 - 1) / (y**2 - 1 - A_b))

tabella = pd.DataFrame({
    "lambda nominale":  x,
    "lambda implicita": lam_implicita,
    "differenza [nm]":  lam_implicita - x,
})
display(tabella.sort_values("lambda nominale").round(0))


