from biocrnpyler import *
import numpy as np     #for arrays
import pylab as plt    #for plotting
import bioscrape as bs #for simulation
import pandas          #stores simulation results

# # This is a TSV that we will need for parameters

# f = open("default_parameters.txt", 'w')
# f.write("""mechanism_id	part_id	param_name	param_val	comments
# 	e coli	protein_Ribo_machinery	150	uM assuming ~100000 Ribosomes / e. coli with a volume 1 um^3
# 	e coli	protein_RNAP_machinery	15	uM assuming ~10000 RNAP molecules / e. coli with a volume 1 um^3
# 	e coli	protein_RNAase_machinery	45	uM assuming ~30000 RNAP molecules / e. coli with a volume 1 um^4
# 	e coli	cellular_processes	5	somewhat arbitary concentration for ~3000 genes in e. coli assuming weak loading on all of them
# 	e coli extract	protein_Ribo	24	1/5 th the Ribosome concentration of E. Coli
# 	e coli extract	protein_RNAP	3	1/5 th the rnap concentration of E. Coli
# 	e coli extract	protein_RNAase	6	1/5 th the rnaase concentration of E. Coli
# 	e coli extract 2	protein_Ribo	12	
# 	e coli extract 2	protein_RNAP	6	
# 	e coli extract 2	protein_RNAase	3	
# 		ktx	0.05	transcripts / second per polymerase assuming 50nt/s and transcript length of 1000
# 		ktl	0.05	proteins / second per ribosome assuming 15aa/s and protein length of 300
# 		cooperativity	2	Seems like a good default
# 		kb	100	assuming 10ms to diffuse across 1um (characteristic cell size)
# 		ku	10	90% binding
# 		n	2	cooperativity is soemtiems called n
# 		kdil	0.001	assuming half life of ~20 minutes for everything (e coli doubling time)
# rna_degredation_mm		kdeg	0.1	assuming a half life of ~10 minutes for mRNA
# rna_degredation		kdil	0.1	assuming a half life of ~10 minutes for mRNA
# simple_transcription		ktx	0.1	Assuming ~10% of e coli rnap working at ktx above
# simple_translation		ktl	0.25	Assuming ~5% of e coli ribosomes working at ktx above
# gene_expression		kexpress	0.28125	The product of the above two rates
# negativehill_transcription		k	0.01875	Estimates from Repressilator paper and the above numbers
# negativehill_transcription		K	20	Estimates from Repressilator paper and the above numbers
# negativehill_transcription		n	2	Estimates from Repressilator paper and the above numbers
# negativehill_transcription		kleak	0.000001	Estimates from Repressilator paper and the above numbers
# transcription	J23103	ku	588.2352941	Anderson promoter (17x) strengths are proportional but gain factor is arbitrary
# transcription	J23116	ku	25.25252525	Anderson promoter (396x) strengths are proportional but gain factor is arbitrary
# transcription	J23107	ku	11.01321586	Anderson promoter (908x) strengths are proportional but gain factor is arbitrary
# transcription	J23106	ku	8.438818565	Anderson promoter (1185x) strengths are proportional but gain factor is arbitrary
# transcription	J23102	ku	4.589261129	Anderson promoter (2179x) strengths are proportional but gain factor is arbitrary
# transcription	J23100	ku	3.926187672	Anderson promoter (2547x) strengths are proportional but gain factor is arbitrary
# simple_transcription	J23103	ktx	0.0031875	Above rescaled by ktx for simple transcription
# simple_transcription	J23116	ktx	0.07425	Above rescaled by ktx for simple transcription
# simple_transcription	J23107	ktx	0.17025	Above rescaled by ktx for simple transcription
# simple_transcription	J23106	ktx	0.2221875	Above rescaled by ktx for simple transcription
# simple_transcription	J23102	ktx	0.4085625	Above rescaled by ktx for simple transcription
# simple_transcription	J23100	ktx	0.4775625	Above rescaled by ktx for simple transcription
# translation	BCD2	ku	0.5	Strong BCD param value made up
# translation	BCD8	ku	10	Weak BCD param value made up
# translation	BCD12	ku	5	medium BCD param value made up
# simple_translation	BCD2	ktl	0.6	Above rescaled by ktl for simple translation
# simple_translation	BCD8	ktl	0.075	Above rescaled by ktl for simple translation
# simple_translation	BCD12	ktl	0.06	Above rescaled by ktl for simple translation
# translation	strong	ku	0.5	Strong BCD param value made up
# translation	medium	ku	5	Weak BCD param value made up
# translation	weak	ku	10	medium BCD param value made up
# translation	very_weak	ku	60	medium BCD param value made up
# simple_translation	strong	ktl	0.6	Above rescaled by ktl for simple translation
# simple_translation	medium	ktl	0.075	Above rescaled by ktl for simple translation
# simple_translation	weak	ktl	0.06	Above rescaled by ktl for simple translation
# simple_translation	very weak	ktl	0.01	Above rescaled by ktl for simple translation
# transcription	weak	ktx	588.2352941	Anderson promoter (17x) strengths are proportional but gain factor is arbitrary
# transcription	medium	ktx	8.438818565	Anderson promoter (1185x) strengths are proportional but gain factor is arbitrary
# transcription	strong	ktx	3.926187672	Anderson promoter (2547x) strengths are proportional but gain factor is arbitrary
# simple_transcription	weak	ktx	0.0031875	Above rescaled by ktx for simple transcription
# simple_transcription	medium	ktx	0.2221875	Above rescaled by ktx for simple transcription
# simple_transcription	strong	ktx	0.4775625	Above rescaled by ktx for simple transcription
# 	combinatorial_promoter_leak	ktx	0.0005	1% of ktx default
# 	combinatorial_promoter_leak	kexpress	0.0028125	1% of kexpress default
# 	regulated_promoter_leak	ktx	0.0005	1% of ktx default
# 	regulated_promoter_leak	kexpress	0.0028125	1% of kexpress default
# deg_tagged_degredation		kdeg	0.1	
# 		kcat	100	use with caution - this depends a lot on what is being catalyzed!""")
# f.close()

#Species
AHL = Species("AHL", material_type = "small_molecule")
LuxR = Species("LuxR", material_type = "Protein")
LuxI = Species("LuxI", material_type = "Protein")
AiiA = Species("AiiA", material_type = "Protein")
sfGFP = Species("sfGFP", material_type = "Protein")
enzyme1= Enzyme(LuxI , substrates = [], products = [AHL])
enzyme2= Enzyme(AiiA , substrates = [AHL], products = [])

params = {"k":1.0, "n":4, "K":20, "kleak":.01}

#Components
activator = ChemicalComplex([LuxR, AHL], "degradable") 
#LuxR Assembly
luxp = ActivatablePromoter(name = "luxp", activator = activator , leak= True, parameters = params)
assembly_luxR = DNAassembly(name   = "LuxR", promoter = luxp , rbs = "strong",protein = LuxR)

#LuxI Assembly
luxp = ActivatablePromoter(name = "luxp", activator = activator , leak= True, parameters = params)
assembly_luxi = DNAassembly(name = "LuxI", promoter = luxp , rbs = "strong", protein = LuxI)

#AiiA Assembly
luxp = ActivatablePromoter(name = "luxp", activator = activator , leak= True, parameters = params)
assembly_aiiA = DNAassembly(name = "AiiA", promoter = luxp , rbs = "medium",  protein = AiiA)

#sfGFP Assembly
luxp = ActivatablePromoter(name = "luxp", activator = activator , leak= True, parameters = params)
assembly_sfgfp = DNAassembly(name = "sfGFP", promoter = luxp, rbs = "strong", protein = sfGFP)

#Create a mixture
mixture = SimpleTxTlExtract("e_coli", components = [enzyme1, enzyme2, assembly_luxi, assembly_luxR, assembly_aiiA, assembly_sfgfp], parameter_file = "default_parameters.txt")
crn = mixture.compile_crn()
print(crn.pretty_print(show_rates = True))


#Create initial conditions
x0 = {str(LuxR):0, str(LuxI):0, str(AiiA):0, str(sfGFP):0}

timepoints = np.linspace(0, 200, 100)
R = crn.simulate_with_bioscrape_via_sbml(timepoints, initial_condition_dict = x0)
plt.plot(timepoints, R[str(LuxR)], label = str(LuxR))
plt.plot(timepoints, R[str(LuxI)], label = str(LuxI))
plt.plot(timepoints, R[str(AiiA)], label = str(AiiA))
plt.plot(timepoints, R[str(AHL)], label = str(AHL))
plt.plot(timepoints, R[str(sfGFP)], label = str(sfGFP))
plt.legend()
plt.show()