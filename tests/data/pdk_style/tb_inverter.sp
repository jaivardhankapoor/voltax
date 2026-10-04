PDK-style testbench: inverter + diode + poly resistor + npn
.lib './models.lib' tt
.param supply=1.8 wn=0.84 wp={2*wn}
Vdd vdd 0 {supply}
Vin in 0 PULSE(0 {supply} 0 20p 20p 1n 2n)
Xn out in 0 0 pdk_nfet_01v8 w={wn} l=0.15 ad={wn*0.29} as={wn*0.29}
+ pd={2*(wn+0.29)} ps={2*(wn+0.29)} sa=0.27 sb=0.27
Xp out in vdd vdd pdk_pfet_01v8 w={wp} l=0.15
Xlong out2 in 0 0 pdk_nfet_01v8 w=5 l=1 mult=2
Rl2 vdd out2 10k
Xr vdd mid pdk_res_high_po w=0.35 l={2*1.5}
D1 0 mid pdk_diode_pw2nd area=2
Q1 vdd mid qe pdk_npn
Re qe 0 1k
.tran 10p 4n
.end
