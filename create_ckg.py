from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from utils import visualize

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
g.add((SNOMED.Patient, SNOMED.hasOutcome, SNOMED.Death))
g.add((SNOMED.Patient, SNOMED.hasOutcome, SNOMED.PatientDischargedAlive))

# Outcomes
g.add((SNOMED.Death, RDFS.subClassOf, SNOMED.Outcome))
g.add((SNOMED.Death, RDF.type, SNOMED.Outcome))
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
# g.add((SNOMED.RiskFactor, RDF.type, OWL.Class))
g.add((SNOMED.Age, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.Age, SCHEMA.greaterOrEqual, Literal(65, datatype=XSD.integer)))
g.add((SNOMED.Age, SCHEMA.increaseRiskOf, SNOMED.Sepsis))

g.add((SNOMED.CReactiveProtein, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.CReactiveProtein, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.CReactiveProtein, SCHEMA.greaterOrEqual, Literal(100, datatype=XSD.integer)))

g.add((SNOMED.Lactate, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.Lactate, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.Lactate, SCHEMA.greaterOrEqual, Literal(4, datatype=XSD.integer)))

g.add((SNOMED.WhiteBloodCells, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.WhiteBloodCells, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.WhiteBloodCells, SCHEMA.greaterOrEqual, Literal(30, datatype=XSD.integer)))

g.add((SNOMED.Bilirubin, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.Bilirubin, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.Bilirubin, SCHEMA.greaterOrEqual, Literal(2, datatype=XSD.float)))

g.add((SNOMED.Platelet, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.Platelet, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.Platelet, SCHEMA.lessOrEqual, Literal(50, datatype=XSD.integer)))

g.add((SNOMED.MeanArterialPressure, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.MeanArterialPressure, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.MeanArterialPressure, SCHEMA.lessOrEqual, Literal(65, datatype=XSD.integer)))

g.add((SNOMED.Cancer, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.Cancer, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.Cancer, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.Pneumonia, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.Pneumonia, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.Pneumonia, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.HIV, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.HIV, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.HIV, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.CirrhosisOfLiver, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.CirrhosisOfLiver, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.CirrhosisOfLiver, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.KidneyDisease, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.KidneyDisease, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.KidneyDisease, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.CardiacInsufficiency, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.CardiacInsufficiency, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.CardiacInsufficiency, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.ChronicDisease, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.ChronicDisease, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.ChronicDisease, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.SepticShock, RDFS.subClassOf, SNOMED.RiskFactor))
g.add((SNOMED.SepticShock, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.Hypotension, RDFS.subClassOf, SNOMED.RiskFactor))
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