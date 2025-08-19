from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from utils import visualize
import networkx as nx

CKG = Namespace("http://example.org/clinical-guideline/")
SNOMED = Namespace("http://snomed.info/id/")
SCHEMA = Namespace("https://schema.org/")

g = Graph()

g.bind("snomed", SNOMED)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)
g.bind("sct", XSD)
g.bind("sct", SNOMED)
g.bind("schema", SCHEMA)

# Patient
#g.add((SNOMED.Patient, RDF.type, OWL.Class))
g.add((SNOMED.Patient, SNOMED.hasOutcome, SNOMED.Death))
g.add((SNOMED.Patient, SNOMED.hasOutcome, SNOMED.PatientDischargedAlive))

# Diseases
# g.add((SNOMED.Cancer, RDF.type, OWL.Class))
# g.add((SNOMED.Pneumonia, RDF.type, OWL.Class))
# g.add((SNOMED.Sepsis, RDF.type, OWL.Class))

# Outcomes
# g.add((SNOMED.Death, RDF.type, OWL.Class))
g.add((SNOMED.Death, RDFS.subClassOf, SNOMED.Outcome))
g.add((SNOMED.Death, RDF.type, SNOMED.Outcome))
# g.add((SNOMED.PatientDischargedAlive, RDF.type, OWL.Class))
g.add((SNOMED.PatientDischargedAlive, RDFS.subClassOf, SNOMED.Outcome))

# Patient attributes
g.add((CKG.hasAge, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasAge, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasAge, RDFS.range, SNOMED.Age))

g.add((CKG.hasBMI, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasBMI, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasBMI, RDFS.range, SNOMED.BodyMassIndex))

g.add((CKG.hasMeanArterialPressure, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasMeanArterialPressure, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasMeanArterialPressure, RDFS.range, SNOMED.MeanArterialPressure))

g.add((CKG.hasSystolicArterialPressure, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasSystolicArterialPressure, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasSystolicArterialPressure, RDFS.range, SNOMED.SystolicArterialPressure))

g.add((CKG.hasDiastolicArterialPressure, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasDiastolicArterialPressure, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasDiastolicArterialPressure, RDFS.range, SNOMED.DiastolicArterialPressure))

g.add((CKG.hasHeartRate, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasHeartRate, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasHeartRate, RDFS.range, SNOMED.HeartRate))

g.add((CKG.hasRespiratoryRate, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasRespiratoryRate, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasRespiratoryRate, RDFS.range, SNOMED.RespiratoryRate))

g.add((CKG.hasTemperature, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasTemperature, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasTemperature, RDFS.range, SNOMED.BodyTemperature))

g.add((CKG.hasOxygenSaturation, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasOxygenSaturation, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasOxygenSaturation, RDFS.range, SNOMED.OxygenSaturation))

# RiskFactor for death
# g.add((SCHEMA.riskFactor, RDF.type, OWL.Class))
g.add((CKG.AgeRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((CKG.AgeRiskFactor, SCHEMA.propertyId, SNOMED.Age))
g.add((CKG.AgeRiskFactor, SNOMED.hasOutcome, SNOMED.Death))
g.add((CKG.AgeRiskFactor, SCHEMA.greaterOrEqual, Literal(65, datatype=XSD.integer)))
g.add((CKG.AgeRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Sepsis))

g.add((CKG.CReactiveProteinRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((CKG.CReactiveProteinRiskFactor, SCHEMA.propertyId, SNOMED.CReactiveProtein))
g.add((CKG.CReactiveProteinRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((CKG.CReactiveProteinRiskFactor, SCHEMA.greaterOrEqual, Literal(100, datatype=XSD.integer)))

g.add((CKG.LactateRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((CKG.LactateRiskFactor, SCHEMA.propertyId, SNOMED.Lactate))
g.add((CKG.LactateRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((CKG.LactateRiskFactor, SCHEMA.greaterOrEqual, Literal(4, datatype=XSD.integer)))

g.add((CKG.WhiteBloodCellsRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((CKG.WhiteBloodCellsRiskFactor, SCHEMA.propertyId, SNOMED.WhiteBloodCells))
g.add((CKG.WhiteBloodCellsRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((CKG.WhiteBloodCellsRiskFactor, SCHEMA.greaterOrEqual, Literal(30, datatype=XSD.integer)))

g.add((CKG.BilirubinRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((CKG.BilirubinRiskFactor, SCHEMA.propertyId, SNOMED.Bilirubin))
g.add((CKG.BilirubinRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((CKG.BilirubinRiskFactor, SCHEMA.greaterOrEqual, Literal(2, datatype=XSD.float)))

g.add((CKG.PlateletRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((CKG.PlateletRiskFactor, SCHEMA.propertyId, SNOMED.Platelet))
g.add((CKG.PlateletRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((CKG.PlateletRiskFactor, SCHEMA.lessOrEqual, Literal(50, datatype=XSD.integer)))

g.add((CKG.MeanArterialPressureRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((CKG.MeanArterialPressureRiskFactor, SCHEMA.propertyId, SNOMED.MeanArterialPressure))
g.add((CKG.MeanArterialPressureRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((CKG.MeanArterialPressureRiskFactor, SCHEMA.lessOrEqual, Literal(65, datatype=XSD.integer)))

g.add((SNOMED.Cancer, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.Cancer, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.Cancer, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.Pneumonia, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.Pneumonia, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.Pneumonia, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.HIV, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.HIV, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.HIV, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.CirrhosisOfLiver, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.CirrhosisOfLiver, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.CirrhosisOfLiver, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.KidneyDisease, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.KidneyDisease, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.KidneyDisease, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.CardiacInsufficiency, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.CardiacInsufficiency, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.CardiacInsufficiency, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.ChronicDisease, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.ChronicDisease, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.ChronicDisease, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.SepticShock, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.SepticShock, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.Hypotension, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.Hypotension, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.Hypotension, SNOMED.causedBy, SNOMED.Sepsis))

g.add((SNOMED.SingleOrganDysfunctionSyndrome, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.SingleOrganDysfunctionSyndrome, SNOMED.causedBy, SNOMED.Sepsis))

g.add((SNOMED.MultipleOrganDysfunctionSyndrome, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.MultipleOrganDysfunctionSyndrome, SNOMED.causedBy, SNOMED.Sepsis))


# RiskFactor for sepsis
g.add((SNOMED.Cancer, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.Pneumonia, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.HIV, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.CirrhosisOfLiver, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.KidneyDisease, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.CardiacInsufficiency, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.SepticShock, SCHEMA.increasesRiskOf, SNOMED.Sepsis))

visualize(g)