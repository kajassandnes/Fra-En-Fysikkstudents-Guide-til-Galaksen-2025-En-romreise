"""IKKE BRUKT SKJELETTKODE <3"""
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
#konstantene
L = 1e-6 #m
N = 100 #antall hydrogenmolekyler i boksen
k = const.k_B #boltzmanns konstant
T = 3*1e3
m = const.m_H2
STD = np.sqrt((k*T) / m) #må endres
r = np.random.uniform(0, L, size=(int(N),3))
v = np.random.normal(0, STD, size=(int(N),3))
eps = 1e-10 #for å sikre kjøringen i loopen
tid = 1e-9 #hvor lenge koden skal kjøre
dt = 1e-12 
steg = 1000 #steg lengde per sek
F = 0
kraft = 0 #For å finne kraften som summeres av trykket til alle partiklene
hbPart = 0 #hadebra partikler som har forlatt gasskammeret
mask = []

#masker for partikler som er utenfor 
"siden hastigheten er konstant trenger vi ikke å integrere, men kun oppdatere hastighetene med s=vt (Sveriges Television)"
"her simulerer vi en fysisk tid på 10 sekunder "


for _ in range(steg):
    r += v * dt

    #lager masker for partiklene som er utenfor veggen
    mask_venstre = r < 0
    mask_høyre = r > L
    for i in range(len(mask_venstre)):
        mask_venstre[i][2] = False
    "Denne masken forteller oss hvilke av partiklene som er utenfor boksen nå som vi har oppdatert posisjonen"
    "mask_høyre returnerer en array med True eller False avhengig av om partikkelen er utenfor"
    "True gir utenfor, False gir inne"
    
    "Ønsker da å endre på hastighetskomponenten (f.eks v_r = -v_r) i den komponenten som inneholder True"
    #v[mask_venstre] = -v[mask_venstre]
    "Når partikkelen går forbi veggen vil vi at den skal gå tilbake innenfor veggen med en avstand som tilsvarer det den dro utenfor L"
    #r[mask_høyre] = L - r[mask_høyre]
    "Oppdaterer hastigheten, hvis høyre eller venstre maske er oversteget L eller 0 vil fartskomponenten flippes"
    #v[mask_venstre | mask_høyre] *= -1
    v[mask_venstre[:,0] | mask_høyre[:,0],0] *= (-1)
    v[mask_venstre[:,1] | mask_høyre[:,1],1] *= (-1)
    v[mask_venstre[:,2] | mask_høyre[:,2],2] *= (-1)
    #--------
    "Nå skal vi introdusere en åpning i boksen"
    "Da må vi at z=0. Da vil partikkelen kunne unnslippe og gi en kraft til raketten"
    
    mask = r[:,2] < 0
    "Nå har vi alle partiklene per gjennomkjøring som har forlatt gasskammeret (boksen) vår"
    "Nå må vi hente ut hastigheten herfra og regne drivet. Som er normalt på utgangsveggen (z-veggen)"
    #v_norm = v[mask2]
    "Alle partiklene som flyr ut skal erstattes av partikler som blir uniformt fordelt i toppen av boksen med tilfeldig hastighet"
    N_out = np.sum(mask)
    v_norm = v[mask, 2]
    #endringen i driv per partikkel
    "partiklene beveger seg i negativ z-retning, derfor minus"
    delta_p = - 2*m *v_norm
    #Kraft på veggen i dette tidssteget
    F += np.sum(delta_p) / dt
    #print(v_norm)
    r[mask] = np.random.uniform([0,0,L-eps], [L,L,L-eps], size=(N_out,3))
    v[mask] = np.random.normal(0, STD, size=(N_out,3))

antallBokser = 2.5e13
rakettmotorAreal = (L**2)*antallBokser
print(f'---------')
print(f'Rakettmotorens areal: {rakettmotorAreal} m^2')
print(f'Mengde kraft fra motor: {F*antallBokser:.1f} N')


#(r[:,0] > 0.25*L) & (r[:,0] < 0.75*L) & #at r-komponent er innenfor hullet i r retning
        #(r[:,1] > 0.25*L) & (r[:,1] < 0.75*L) & #at y-komponent er innenfor hullet i y retning













